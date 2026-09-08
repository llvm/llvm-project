# Generates the switch-fragment headers consumed by the libherbceptions
# runtime sources. Each fragment is a switch body included inside a
# per-domain do_to_u8scatter / do_to_errc function; the generator owns the
# data, the generated files must not be edited by hand.
#
# Outputs (all under src/):
#
#   posix_table.hpp       posix.cpp       errno -> message scatter
#   parse_table.hpp       parse.cpp       parse_errc -> message scatter
#   cmath_table.hpp       cmath_errc.cpp  cmath_errc -> message scatter
#   wine_table.hpp        wine.cpp        wine errno -> message scatter
#   wine_errc_map.hpp     wine.cpp        wine_errc -> std::errc
#   win32_errc_map.hpp    win32.cpp       win32 GetLastError() -> std::errc
#   nt_errc_map.hpp       nt.cpp          failed NTSTATUS -> std::errc
#   nt_message_table.hpp  nt.cpp (ntkernel.h)  NTSTATUS -> message scatter
#   nt_message_max.hpp    nt.cpp (ntkernel.h)  max message length
#
# Data sources:
#
#   * The POSIX / parse / cmath / wine tables are hard-coded below.
#   * The win32 / nt / nt-message tables are read from
#     utils/ntkernel-table.json (vendored from ntkernel-error-category,
#     Apache-2.0 / Boost-1.0, Niall Douglas).

import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "..", "src")
JSON_PATH = os.path.join(SCRIPT_DIR, "ntkernel-table.json")

# ---------------------------------------------------------------------------
# POSIX (std::errc) table
# ---------------------------------------------------------------------------

ERRORS = [
    ("0", "Success"),

    ("EPERM", "Not owner",
        "defined(EPERM) && (!defined(EACCES) || (EPERM != EACCES))"),

    ("ENOENT", "No such file or directory"),

    ("ESRCH", "No such process"),
    ("EINTR", "Interrupted system call"),
    ("EIO", "I/O error"),

    ("ENXIO", "No such device or address",
        "defined(ENXIO) && (!defined(ENODEV) || (ENXIO != ENODEV))"),

    ("E2BIG", "Arg list too long"),
    ("ENOEXEC", "Exec format error"),
    ("EALREADY", "Socket already connected"),
    ("EBADF", "Bad file number"),
    ("ECHILD", "No children"),
    ("EDESTADDRREQ", "Destination address required"),
    ("EAGAIN", "No more processes"),
    ("ENOMEM", "Not enough space"),
    ("EACCES", "Permission denied"),
    ("EFAULT", "Bad address"),
    ("ENOTBLK", "Block device required"),
    ("EBUSY", "Device or resource busy"),
    ("EEXIST", "File exists"),
    ("EXDEV", "Cross-device link"),
    ("ENODEV", "No such device"),
    ("ENOTDIR", "Not a directory"),
    ("EHOSTDOWN", "Host is down"),
    ("EINPROGRESS", "Connection already in progress"),
    ("EISDIR", "Is a directory"),
    ("EINVAL", "Invalid argument"),
    ("ENETDOWN", "Network interface is not configured"),
    ("ENETRESET", "Connection aborted by network"),
    ("ENFILE", "Too many open files in system"),
    ("EMFILE", "File descriptor value too large"),
    ("ENOTTY", "Not a character device"),
    ("ETXTBSY", "Text file busy"),
    ("EFBIG", "File too large"),
    ("EHOSTUNREACH", "Host is unreachable"),
    ("ENOSPC", "No space left on device"),
    ("ENOTSUP", "Not supported"),
    ("ESPIPE", "Illegal seek"),
    ("EROFS", "Read-only file system"),
    ("EMLINK", "Too many links"),
    ("EPIPE", "Broken pipe"),
    ("EDOM", "Mathematics argument out of domain of function"),
    ("ERANGE", "Result too large"),
    ("ENOMSG", "No message of desired type"),
    ("EIDRM", "Identifier removed"),
    ("EILSEQ", "Illegal byte sequence"),
    ("EDEADLK", "Deadlock"),
    ("ENETUNREACH", "Network is unreachable"),
    ("ENOLCK", "No lock"),
    ("ENOSTR", "Not a stream"),
    ("ETIME", "Stream ioctl timeout"),
    ("ENOSR", "No stream resources"),
    ("ENONET", "Machine is not on the network"),
    ("ENOPKG", "No package"),
    ("EREMOTE", "Resource is remote"),
    ("ENOLINK", "Virtual circuit is gone"),
    ("EADV", "Advertise error"),
    ("ESRMNT", "Srmount error"),
    ("ECOMM", "Communication error"),
    ("EPROTO", "Protocol error"),
    ("EPROTONOSUPPORT", "Unknown protocol"),
    ("EMULTIHOP", "Multihop attempted"),
    ("EBADMSG", "Bad message"),
    ("ELIBACC", "Cannot access a needed shared library"),
    ("ELIBBAD", "Accessing a corrupted shared library"),
    ("ELIBSCN", ".lib section in a.out corrupted"),
    ("ELIBMAX", "Attempting to link in more shared libraries than system limit"),
    ("ELIBEXEC", "Cannot exec a shared library directly"),
    ("ENOSYS", "Function not implemented"),
    ("ENMFILE", "No more files"),
    ("ENOTEMPTY", "Directory not empty"),
    ("ENAMETOOLONG", "File or path name too long"),
    ("ELOOP", "Too many symbolic links"),
    ("ENOBUFS", "No buffer space available"),
    ("ENODATA", "No data"),
    ("EAFNOSUPPORT", "Address family not supported by protocol family"),
    ("EPROTOTYPE", "Protocol wrong type for socket"),
    ("ENOTSOCK", "Socket operation on non-socket"),
    ("ENOPROTOOPT", "Protocol not available"),
    ("ESHUTDOWN", "Can't send after socket shutdown"),
    ("ECONNREFUSED", "Connection refused"),
    ("ECONNRESET", "Connection reset by peer"),
    ("EADDRINUSE", "Address already in use"),
    ("EADDRNOTAVAIL", "Address not available"),
    ("ECONNABORTED", "Software caused connection abort"),

    ("EWOULDBLOCK", "Operation would block",
        "(defined(EWOULDBLOCK) && (!defined(EAGAIN) || (EWOULDBLOCK != EAGAIN)))"),

    ("ENOTCONN", "Socket is not connected"),
    ("ESOCKTNOSUPPORT", "Socket type not supported"),
    ("EISCONN", "Socket is already connected"),
    ("ECANCELED", "Operation canceled"),
    ("ENOTRECOVERABLE", "State not recoverable"),
    ("EOWNERDEAD", "Previous owner died"),
    ("ESTRPIPE", "Streams pipe error"),

    ("EOPNOTSUPP", "Operation not supported on socket",
        "defined(EOPNOTSUPP) && (!defined(ENOTSUP) || (ENOTSUP != EOPNOTSUPP))"),

    ("EOVERFLOW", "Value too large for defined data type"),
    ("EMSGSIZE", "Message too long"),
    ("ETIMEDOUT", "Connection timed out"),

    ("UNKNOWN", "Unknown")
]

COMMENTS = {
    "ENXIO": "/* go32 defines ENXIO as ENODEV */",
}


def max_message_size() -> int:
    return max(len(msg) for _, msg, *_ in ERRORS)


def emit_posix() -> str:
    lines = ["// clang-format off", f"#define POSIX_ERRC_MAX_SIZE {max_message_size()}", ""]
    for name, msg, *rest in ERRORS:
        if name == "0":
            lines.append("\tcase 0:")
        elif name == "UNKNOWN":
            lines.append("\tdefault:")
        else:
            if name in COMMENTS:
                lines.append(COMMENTS[name])
            if rest:
                lines.append(f"#if {rest[0]}")
            else:
                lines.append(f"#ifdef {name}")
            lines.append(f"\tcase {name}:")
        lines.append(f"\t\treturn __tsc(u8\"{msg}\");")
        if name not in ("0", "UNKNOWN"):
            lines.append("#endif")
    lines.append("// clang-format on")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Generic fragment writer, shared by the parse / cmath / wine tables.
# ---------------------------------------------------------------------------

def write_fragment(path: str, max_macro: str, rows, default_msg: str) -> None:
    all_msgs = [msg for _, msg in rows] + [default_msg]
    lines = ["// clang-format off", f"#define {max_macro} {max(len(m) for m in all_msgs)}", ""]
    wrote_default = False
    for label, msg in rows:
        if label is None:
            lines.append("\tdefault:")
            wrote_default = True
        else:
            lines.append(f"\tcase {label}:")
        lines.append(f"\t\treturn __tsc(u8\"{msg}\");")
    if not wrote_default:
        lines.append("\tdefault:")
        lines.append(f"\t\treturn __tsc(u8\"{default_msg}\");")
    lines.append("// clang-format on")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.abspath(path)}")


# ---------------------------------------------------------------------------
# parse_errc table
# ---------------------------------------------------------------------------

PARSE_ERRORS = [
    (0, "Success"),
    (1, "End of file"),
    (2, "Partial parse"),
    (3, "Invalid format"),
    (4, "Overflow"),
]


# ---------------------------------------------------------------------------
# cmath_errc table
# ---------------------------------------------------------------------------

CMATH_ERRORS = [
    ("invalid", "Invalid floating point operation"),
    ("divbyzero", "Floating point divide by zero"),
    ("inexact", "Inexact floating point result"),
    ("overflow", "Floating point overflow"),
    ("underflow", "Floating point underflow"),
    ("all_except", "All floating point exceptions"),
]


def emit_cmath() -> str:
    default_msg = "Unknown"
    msgs = [msg for _, msg in CMATH_ERRORS] + [default_msg]
    lines = [
        "// clang-format off",
        "// Requires herbceptions/__details/cmath_errc.h (defines ::std::cmath_errc).",
        f"#define CMATH_ERRC_MAX_SIZE {max(len(m) for m in msgs)}",
        "",
    ]
    for name, msg in CMATH_ERRORS:
        lines.append(f"\tcase static_cast<::std::uint_least32_t>(::std::cmath_errc::{name}):")
        lines.append(f"\t\treturn __tsc(u8\"{msg}\");")
    lines.append("\tdefault:")
    lines.append(f"\t\treturn __tsc(u8\"{default_msg}\");")
    lines.append("// clang-format on")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# wine errno / wine_errc_map tables
# ---------------------------------------------------------------------------

WINE_ERRNO_HEADER = os.path.join(SCRIPT_DIR, "__wine_unix_errno.h")

WINE_ERRC_MAP_NAMES = {
    "EPERM": ("operation_not_permitted", None),
    "ENOENT": ("no_such_file_or_directory", None),
    "ESRCH": ("no_such_process", None),
    "EINTR": ("interrupted", None),
    "EIO": ("io_error", None),
    "ENXIO": ("no_such_device_or_address", None),
    "E2BIG": ("argument_list_too_long", None),
    "ENOEXEC": ("executable_format_error", None),
    "EBADF": ("bad_file_descriptor", None),
    "ECHILD": ("no_child_process", None),
    "EAGAIN": ("resource_unavailable_try_again", None),
    "ENOMEM": ("not_enough_memory", None),
    "EFAULT": ("bad_address", None),
    "EBUSY": ("device_or_resource_busy", None),
    "EEXIST": ("file_exists", None),
    "EXDEV": ("cross_device_link", None),
    "ENODEV": ("no_such_device", None),
    "ENOTDIR": ("not_a_directory", None),
    "EISDIR": ("is_a_directory", None),
    "EINVAL": ("invalid_argument", None),
    "ENFILE": ("too_many_files_open_in_system", None),
    "EMFILE": ("too_many_files_open", None),
    "ENOTTY": ("inappropriate_io_control_operation", None),
    "ETXTBSY": ("text_file_busy", None),
    "EFBIG": ("file_too_large", None),
    "EHOSTUNREACH": ("host_unreachable", None),
    "ENOSPC": ("no_space_on_device", None),
    "ENOTSUP": ("operation_not_supported", None),
    "ESPIPE": ("invalid_seek", None),
    "EROFS": ("read_only_file_system", None),
    "EMLINK": ("too_many_links", None),
    "EPIPE": ("broken_pipe", None),
    "EDOM": ("argument_out_of_domain", None),
    "ERANGE": ("result_out_of_range", None),
    "ENOMSG": ("no_message", None),
    "EIDRM": ("identifier_removed", None),
    "EILSEQ": ("illegal_byte_sequence", None),
    "EDEADLK": ("resource_deadlock_would_occur", None),
    "ENETUNREACH": ("network_unreachable", None),
    "ENOLCK": ("no_lock_available", None),
    "ENOLINK": ("no_link", None),
    "EPROTO": ("protocol_error", None),
    "EPROTONOSUPPORT": ("protocol_not_supported", None),
    "EBADMSG": ("bad_message", None),
    "ENOSYS": ("function_not_supported", None),
    "ENOTEMPTY": ("directory_not_empty", None),
    "ENAMETOOLONG": ("filename_too_long", None),
    "ELOOP": ("too_many_symbolic_link_levels", None),
    "ENOBUFS": ("no_buffer_space", None),
    "EAFNOSUPPORT": ("address_family_not_supported", None),
    "EPROTOTYPE": ("wrong_protocol_type", None),
    "ENOTSOCK": ("not_a_socket", None),
    "ENOPROTOOPT": ("no_protocol_option", None),
    "ECONNREFUSED": ("connection_refused", None),
    "ECONNRESET": ("connection_reset", None),
    "EADDRINUSE": ("address_in_use", None),
    "EADDRNOTAVAIL": ("address_not_available", None),
    "ECONNABORTED": ("connection_aborted", None),
    "EWOULDBLOCK": ("would_block", "EAGAIN"),
    "ENOTCONN": ("not_connected", None),
    "EISCONN": ("already_connected", None),
    "ECANCELED": ("operation_canceled", None),
    "ENOTRECOVERABLE": ("state_not_recoverable", None),
    "EOWNERDEAD": ("owner_dead", None),
    "ESTALE": ("stale_file_handle", "ESTALE"),
    "EOVERFLOW": ("value_too_large", None),
    "EMSGSIZE": ("message_size", None),
    "ETIMEDOUT": ("timed_out", None),
}


def emit_wine_errc_map() -> str:
    lines = [
        "// clang-format off",
        "// Requires <cerrno> and herbceptions/__details/wine.h.",
        "",
        "\tcase static_cast<::std::uint_least32_t>(::std::wine_errc::success):",
        "\t\treturn ::std::errc{};",
    ]
    for name, msg, *rest in ERRORS:
        if name in ("0", "UNKNOWN"):
            continue
        mapped = WINE_ERRC_MAP_NAMES.get(name)
        if mapped is None:
            continue
        wine_name, passthrough = mapped
        if rest:
            lines.append(f"#if {rest[0]}")
        else:
            lines.append(f"#ifdef {name}")
        lines.append(f"\tcase static_cast<::std::uint_least32_t>(::std::wine_errc::{wine_name}):")
        if passthrough is None:
            lines.append(f"\t\treturn ::std::errc::{wine_name};")
        else:
            lines.append(f"\t\treturn static_cast<::std::errc>({passthrough});")
        lines.append("#endif")
    lines.append("#ifdef ESTALE")
    lines.append("\tcase static_cast<::std::uint_least32_t>(::std::wine_errc::stale_file_handle):")
    lines.append("\t\treturn static_cast<::std::errc>(ESTALE);")
    lines.append("#endif")
    lines.append("\tdefault:")
    lines.append("\t\treturn ::std::errc::io_error;")
    lines.append("// clang-format on")
    return "\n".join(lines) + "\n"


_WINE_ROW_RE = re.compile(
    r"^#define\s+__WINE_UNIX_ERRNO_([A-Z0-9_]+)\s+(\d+)\s*(?:/\*\s*(.+?)\s*\*/)?\s*$"
)


def wine_rows():
    rows = []
    with open(WINE_ERRNO_HEADER, "r") as f:
        for line in f:
            m = _WINE_ROW_RE.match(line)
            if not m:
                continue
            name, num, msg = m.group(1), int(m.group(2)), m.group(3)
            if msg is None:
                msg = name.replace("_", " ").capitalize()
            rows.append((num, msg))
    rows.sort(key=lambda r: r[0])
    return rows


# ---------------------------------------------------------------------------
# win32 / nt / nt-message tables (from ntkernel-table.json)
# ---------------------------------------------------------------------------

ERRNO_TO_ERRC = {
    "EPERM": "operation_not_permitted",
    "ENOENT": "no_such_file_or_directory",
    "ESRCH": "no_such_process",
    "EINTR": "interrupted",
    "EIO": "io_error",
    "ENXIO": "no_such_device_or_address",
    "E2BIG": "argument_list_too_long",
    "ENOEXEC": "executable_format_error",
    "EBADF": "bad_file_descriptor",
    "ECHILD": "no_child_process",
    "EAGAIN": "resource_unavailable_try_again",
    "ENOMEM": "not_enough_memory",
    "EACCES": "permission_denied",
    "EFAULT": "bad_address",
    "EBUSY": "device_or_resource_busy",
    "EEXIST": "file_exists",
    "EXDEV": "cross_device_link",
    "ENODEV": "no_such_device",
    "ENOTDIR": "not_a_directory",
    "EISDIR": "is_a_directory",
    "EINVAL": "invalid_argument",
    "ENFILE": "too_many_files_open_in_system",
    "EMFILE": "too_many_files_open",
    "ENOTTY": "inappropriate_io_control_operation",
    "ETXTBSY": "text_file_busy",
    "EFBIG": "file_too_large",
    "ENOSPC": "no_space_on_device",
    "ESPIPE": "invalid_seek",
    "EROFS": "read_only_file_system",
    "EMLINK": "too_many_links",
    "EPIPE": "broken_pipe",
    "EDOM": "argument_out_of_domain",
    "ERANGE": "result_out_of_range",
    "EDEADLK": "resource_deadlock_would_occur",
    "ENAMETOOLONG": "filename_too_long",
    "ENOLCK": "no_lock_available",
    "ENOSYS": "function_not_supported",
    "ENOTEMPTY": "directory_not_empty",
    "ELOOP": "too_many_symbolic_link_levels",
    "EOVERFLOW": "value_too_large",
    "EWOULDBLOCK": "operation_would_block",
    "EMSGSIZE": "message_size",
    "EPROTOTYPE": "wrong_protocol_type",
    "ENOPROTOOPT": "no_protocol_option",
    "EPROTONOSUPPORT": "protocol_not_supported",
    "EOPNOTSUPP": "operation_not_supported",
    "ENOTSUP": "operation_not_supported",
    "EAFNOSUPPORT": "address_family_not_supported",
    "EADDRINUSE": "address_in_use",
    "EADDRNOTAVAIL": "address_not_available",
    "ENETDOWN": "network_down",
    "ENETUNREACH": "network_unreachable",
    "ENETRESET": "network_reset",
    "ECONNABORTED": "connection_aborted",
    "ECONNRESET": "connection_reset",
    "EISCONN": "already_connected",
    "ENOTCONN": "not_connected",
    "ETIMEDOUT": "timed_out",
    "ECONNREFUSED": "connection_refused",
    "EHOSTUNREACH": "host_unreachable",
    "EALREADY": "connection_already_in_progress",
    "EINPROGRESS": "operation_in_progress",
    "EDESTADDRREQ": "destination_address_required",
    "ENOBUFS": "no_buffer_space",
    "ENOTSOCK": "not_a_socket",
    "EDQUOT": None,
    "ESTALE": None,
    "ESHUTDOWN": None,
    "EHOSTDOWN": None,
    "EREMOTE": None,
    "EUSERS": None,
    "ECOMM": "protocol_error",
    "EPROTO": "protocol_error",
    "EBADMSG": "bad_message",
    "EILSEQ": "illegal_byte_sequence",
    "ECANCELED": "operation_canceled",
    "ENOTRECOVERABLE": "state_not_recoverable",
    "EOWNERDEAD": "owner_dead",
    "ENOMSG": "no_message",
    "EIDRM": "identifier_removed",
    "ENODATA": "no_message_available",
    "ENOSR": "no_stream_resources",
    "ETIME": "stream_timeout",
}


def load_ntkernel_json():
    with open(JSON_PATH, "r") as f:
        return json.load(f)


def build_nt_cases(data):
    """NT_STATUS -> errno from the JSON's nt + posix columns."""
    cases = {}
    for r in data:
        nt = int(r["nt"], 16)
        posix = r.get("posix")
        if posix:
            cases[nt] = posix
    return cases


def build_win32_cases(data):
    """win32 -> errno from the JSON's win32 + posix columns."""
    cases = {}
    for r in data:
        win32 = int(r["win32"], 16)
        posix = r.get("posix")
        if win32 != 0 and posix:
            cases.setdefault(win32, posix)
    return cases


def emit_errc_map(path, header_note, cases, success_case):
    lines = [
        "// clang-format off",
        "// Generated by utils/generate_table.py; do not edit.",
        "// " + header_note,
        "// Requires <cerrno> and ::std::errc.",
        "",
    ]
    if success_case:
        lines += ["\tcase 0u:", "\t\treturn ::std::errc{};", ""]
    # Group consecutive codes that map to the same return value
    sorted_items = sorted(cases.items())
    groups = []
    current_group = None
    for code, errno in sorted_items:
        errc = ERRNO_TO_ERRC.get(errno, "__passthrough__")
        if errc == "__passthrough__":
            ret = f"static_cast<::std::errc>({errno})"
        else:
            ret = f"::std::errc::{errc}"
        if current_group and current_group["errno"] == errno and current_group["ret"] == ret:
            current_group["codes"].append(code)
        else:
            current_group = {"errno": errno, "ret": ret, "codes": [code]}
            groups.append(current_group)
    for group in groups:
        errno = group["errno"]
        ret = group["ret"]
        lines.append(f"#ifdef {errno}")
        for code in group["codes"]:
            lines.append(f"\tcase {hex(code)}u:")
        lines.append(f"\t\treturn {ret};")
        lines.append("#endif")
    lines += ["", "\tdefault:", "\t\treturn ::std::errc::io_error;",
              "// clang-format on"]
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.abspath(path)} ({len(cases)} mapped codes)")


def emit_nt_message_table(path, data):
    max_len = 0
    lines = [
        "// clang-format off",
        "// Generated from utils/ntkernel-table.json; do not edit.",
        "// NTSTATUS -> US-English UTF-8 message scatter. Requires <herbceptions/error>.",
    ]
    # Group consecutive codes with the same message
    groups = []
    current = None
    for r in data:
        nt = int(r["nt"], 16)
        msg = r.get("message", "")
        if len(msg) > max_len:
            max_len = len(msg)
        if current and current["msg"] == msg:
            current["codes"].append(nt)
        else:
            current = {"msg": msg, "codes": [nt]}
            groups.append(current)
    for group in groups:
        for code in group["codes"]:
            lines.append(f"\tcase {hex(code)}u:")
        escaped = group["msg"].replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        lines.append(f'\t\treturn __tsc(u8"{escaped}");')
    lines.append("// clang-format on")
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.abspath(path)}")
    return max_len


def emit_nt_message_max(path, max_len):
    lines = [
        "// Generated from utils/ntkernel-table.json; do not edit.",
        f"inline constexpr ::std::size_t __nt_max_message_size{{{max_len}}};",
    ]
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.abspath(path)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # posix_table.hpp
    content = emit_posix()
    with open(os.path.join(SRC_DIR, "posix_table.hpp"), "w", newline="\n") as f:
        f.write(content)
    print(f"wrote {os.path.abspath(os.path.join(SRC_DIR, 'posix_table.hpp'))}")

    # parse_table.hpp
    write_fragment(os.path.join(SRC_DIR, "parse_table.hpp"), "PARSE_ERRC_MAX_SIZE",
                   PARSE_ERRORS, "Unknown")

    # cmath_table.hpp
    with open(os.path.join(SRC_DIR, "cmath_table.hpp"), "w", newline="\n") as f:
        f.write(emit_cmath())
    print(f"wrote {os.path.abspath(os.path.join(SRC_DIR, 'cmath_table.hpp'))}")

    # wine_table.hpp + wine_errc_map.hpp
    wine = wine_rows()
    write_fragment(os.path.join(SRC_DIR, "wine_table.hpp"), "WINE_ERRC_MAX_SIZE",
                   wine, "Unknown")
    with open(os.path.join(SRC_DIR, "wine_errc_map.hpp"), "w", newline="\n") as f:
        f.write(emit_wine_errc_map())
    print(f"wrote {os.path.abspath(os.path.join(SRC_DIR, 'wine_errc_map.hpp'))}")

    # win32 / nt / nt-message tables from JSON
    data = load_ntkernel_json()

    win32_cases = build_win32_cases(data)
    nt_cases = build_nt_cases(data)

    unknown = {e for e in list(nt_cases.values()) + list(win32_cases.values())
               if e not in ERRNO_TO_ERRC}
    if unknown:
        print(f"warning: errnos without a known std::errc enumerator "
              f"(emitted as static_cast passthroughs): {sorted(unknown)}",
              file=sys.stderr)

    emit_errc_map(os.path.join(SRC_DIR, "win32_errc_map.hpp"),
                  "win32 GetLastError() error code -> std::errc; case 0 is "
                  "ERROR_SUCCESS.", win32_cases, success_case=True)
    emit_errc_map(os.path.join(SRC_DIR, "nt_errc_map.hpp"),
                  "failed NTSTATUS (severity bits set) -> std::errc; success "
                  "codes are handled by the caller.", nt_cases,
                  success_case=False)

    max_len = emit_nt_message_table(os.path.join(SRC_DIR, "nt_message_table.hpp"), data)
    emit_nt_message_max(os.path.join(SRC_DIR, "nt_message_max.hpp"), max_len)


if __name__ == "__main__":
    main()

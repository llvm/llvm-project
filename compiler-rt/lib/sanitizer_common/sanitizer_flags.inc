//===-- sanitizer_flags.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes common flags available in all sanitizers.
//
//===----------------------------------------------------------------------===//

#ifndef COMMON_FLAG
#error "Define COMMON_FLAG prior to including this file!"
#endif

// COMMON_FLAG(Type, Name, DefaultValue, Description)
// Supported types: bool, const char *, int, uptr.
// Default value must be a compile-time constant.
// Description must be a string literal.

COMMON_FLAG(
    bool, symbolize, true,
    "If set, use the online symbolizer from common sanitizer runtime to turn "
    "virtual addresses to file/line locations.")
COMMON_FLAG(
    const char *, external_symbolizer_path, nullptr,
    "Path to external symbolizer. If empty, the tool will search $PATH for "
    "the symbolizer.")
COMMON_FLAG(
    bool, allow_addr2line, false,
    "If set, allows online symbolizer to run addr2line binary to symbolize "
    "stack traces (addr2line will only be used if llvm-symbolizer binary is "
    "unavailable.")
COMMON_FLAG(const char *, strip_path_prefix, "",
            "Strips this prefix from file paths in error reports.")
COMMON_FLAG(bool, fast_unwind_on_check, false,
            "If available, use the fast frame-pointer-based unwinder on "
            "internal CHECK failures.")
COMMON_FLAG(bool, fast_unwind_on_fatal, false,
            "If available, use the fast frame-pointer-based unwinder on fatal "
            "errors.")
COMMON_FLAG(bool, fast_unwind_on_malloc, true,
            "If available, use the fast frame-pointer-based unwinder on "
            "malloc/free.")
COMMON_FLAG(bool, handle_ioctl, false, "Intercept and handle ioctl requests.")
COMMON_FLAG(int, malloc_context_size, 1,
            "Max number of stack frames kept for each allocation/deallocation.")
COMMON_FLAG(
    const char *, log_path, "stderr",
    "Write logs to \"log_path.pid\". The special values are \"stdout\" and "
    "\"stderr\". The default is \"stderr\".")
COMMON_FLAG(
    bool, log_exe_name, false,
    "Mention name of executable when reporting error and "
    "append executable name to logs (as in \"log_path.exe_name.pid\").")
COMMON_FLAG(
    bool, log_to_syslog, SANITIZER_ANDROID || SANITIZER_MAC,
    "Write all sanitizer output to syslog in addition to other means of "
    "logging.")
COMMON_FLAG(
    int, verbosity, 0,
    "Verbosity level (0 - silent, 1 - a bit of output, 2+ - more output).")
COMMON_FLAG(bool, detect_leaks, true, "Enable memory leak detection.")
COMMON_FLAG(
    bool, leak_check_at_exit, true,
    "Invoke leak checking in an atexit handler. Has no effect if "
    "detect_leaks=false, or if __lsan_do_leak_check() is called before the "
    "handler has a chance to run.")
COMMON_FLAG(bool, allocator_may_return_null, false,
            "If false, the allocator will crash instead of returning 0 on "
            "out-of-memory.")
COMMON_FLAG(bool, print_summary, true,
            "If false, disable printing error summaries in addition to error "
            "reports.")
COMMON_FLAG(bool, check_printf, true, "Check printf arguments.")
COMMON_FLAG(bool, handle_segv, SANITIZER_NEEDS_SEGV,
            "If set, registers the tool's custom SIGSEGV/SIGBUS handler.")
COMMON_FLAG(bool, handle_abort, false,
            "If set, registers the tool's custom SIGABRT handler.")
COMMON_FLAG(bool, handle_sigill, false,
            "If set, registers the tool's custom SIGILL handler.")
COMMON_FLAG(bool, handle_sigfpe, true,
            "If set, registers the tool's custom SIGFPE handler.")
COMMON_FLAG(bool, allow_user_segv_handler, false,
            "If set, allows user to register a SEGV handler even if the tool "
            "registers one.")
COMMON_FLAG(bool, use_sigaltstack, true,
            "If set, uses alternate stack for signal handling.")
COMMON_FLAG(bool, detect_deadlocks, false,
            "If set, deadlock detection is enabled.")
COMMON_FLAG(
    uptr, clear_shadow_mmap_threshold, 64 * 1024,
    "Large shadow regions are zero-filled using mmap(NORESERVE) instead of "
    "memset(). This is the threshold size in bytes.")
COMMON_FLAG(const char *, color, "auto",
            "Colorize reports: (always|never|auto).")
COMMON_FLAG(
    bool, legacy_pthread_cond, false,
    "Enables support for dynamic libraries linked with libpthread 2.2.5.")
COMMON_FLAG(bool, intercept_tls_get_addr, false, "Intercept __tls_get_addr.")
COMMON_FLAG(bool, help, false, "Print the flag descriptions.")
COMMON_FLAG(uptr, mmap_limit_mb, 0,
            "Limit the amount of mmap-ed memory (excluding shadow) in Mb; "
            "not a user-facing flag, used mosly for testing the tools")
COMMON_FLAG(uptr, hard_rss_limit_mb, 0,
            "Hard RSS limit in Mb."
            " If non-zero, a background thread is spawned at startup"
            " which periodically reads RSS and aborts the process if the"
            " limit is reached")
COMMON_FLAG(uptr, soft_rss_limit_mb, 0,
            "Soft RSS limit in Mb."
            " If non-zero, a background thread is spawned at startup"
            " which periodically reads RSS. If the limit is reached"
            " all subsequent malloc/new calls will fail or return NULL"
            " (depending on the value of allocator_may_return_null)"
            " until the RSS goes below the soft limit."
            " This limit does not affect memory allocations other than"
            " malloc/new.")
COMMON_FLAG(bool, can_use_proc_maps_statm, true,
            "If false, do not attempt to read /proc/maps/statm."
            " Mostly useful for testing sanitizers.")
COMMON_FLAG(
    bool, coverage, false,
    "If set, coverage information will be dumped at program shutdown (if the "
    "coverage instrumentation was enabled at compile time).")
COMMON_FLAG(bool, coverage_pcs, true,
            "If set (and if 'coverage' is set too), the coverage information "
            "will be dumped as a set of PC offsets for every module.")
COMMON_FLAG(bool, coverage_order_pcs, false,
             "If true, the PCs will be dumped in the order they've"
             " appeared during the execution.")
COMMON_FLAG(bool, coverage_bitset, false,
            "If set (and if 'coverage' is set too), the coverage information "
            "will also be dumped as a bitset to a separate file.")
COMMON_FLAG(bool, coverage_counters, false,
            "If set (and if 'coverage' is set too), the bitmap that corresponds"
            " to coverage counters will be dumped.")
COMMON_FLAG(bool, coverage_direct, SANITIZER_ANDROID,
            "If set, coverage information will be dumped directly to a memory "
            "mapped file. This way data is not lost even if the process is "
            "suddenly killed.")
COMMON_FLAG(const char *, coverage_dir, ".",
            "Target directory for coverage dumps. Defaults to the current "
            "directory.")
COMMON_FLAG(bool, coverage_pc_buffer, true,
            "If set (and if 'coverage' is set too), the pcs would be collected "
            "in a buffer.")
COMMON_FLAG(bool, full_address_space, false,
            "Sanitize complete address space; "
            "by default kernel area on 32-bit platforms will not be sanitized")
COMMON_FLAG(bool, print_suppressions, true,
            "Print matched suppressions at exit.")
COMMON_FLAG(
    bool, disable_coredump, (SANITIZER_WORDSIZE == 64),
    "Disable core dumping. By default, disable_core=1 on 64-bit to avoid "
    "dumping a 16T+ core file. Ignored on OSes that don't dump core by"
    "default and for sanitizers that don't reserve lots of virtual memory.")
COMMON_FLAG(bool, use_madv_dontdump, true,
          "If set, instructs kernel to not store the (huge) shadow "
          "in core file.")
COMMON_FLAG(bool, symbolize_inline_frames, true,
            "Print inlined frames in stacktraces. Defaults to true.")
COMMON_FLAG(bool, symbolize_vs_style, false,
            "Print file locations in Visual Studio style (e.g: "
            " file(10,42): ...")
COMMON_FLAG(const char *, stack_trace_format, "DEFAULT",
            "Format string used to render stack frames. "
            "See sanitizer_stacktrace_printer.h for the format description. "
            "Use DEFAULT to get default format.")
COMMON_FLAG(bool, no_huge_pages_for_shadow, true,
            "If true, the shadow is not allowed to use huge pages. ")
COMMON_FLAG(bool, strict_string_checks, false,
            "If set check that string arguments are properly null-terminated")
COMMON_FLAG(bool, intercept_strstr, true,
            "If set, uses custom wrappers for strstr and strcasestr functions "
            "to find more errors.")
COMMON_FLAG(bool, intercept_strspn, true,
            "If set, uses custom wrappers for strspn and strcspn function "
            "to find more errors.")
COMMON_FLAG(bool, intercept_strpbrk, true,
            "If set, uses custom wrappers for strpbrk function "
            "to find more errors.")
COMMON_FLAG(bool, intercept_memcmp, true,
            "If set, uses custom wrappers for memcmp function "
            "to find more errors.")
COMMON_FLAG(bool, strict_memcmp, true,
          "If true, assume that memcmp(p1, p2, n) always reads n bytes before "
          "comparing p1 and p2.")
COMMON_FLAG(bool, decorate_proc_maps, false, "If set, decorate sanitizer "
                                             "mappings in /proc/self/maps with "
                                             "user-readable names")
COMMON_FLAG(int, exitcode, 1, "Override the program exit status if the tool "
                              "found an error")
COMMON_FLAG(
    bool, abort_on_error, SANITIZER_MAC,
    "If set, the tool calls abort() instead of _exit() after printing the "
    "error report.")
COMMON_FLAG(bool, suppress_equal_pcs, true,
            "Deduplicate multiple reports for single source location in "
            "halt_on_error=false mode (asan only).")
COMMON_FLAG(bool, print_cmdline, false, "Print command line on crash "
            "(asan only).")
COMMON_FLAG(bool, html_cov_report, false, "Generate html coverage report.")
COMMON_FLAG(const char *, sancov_path, "sancov", "Sancov tool location.")

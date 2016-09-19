//===-- sanitizer_common.cc -----------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is shared between sanitizers' run-time libraries.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_stacktrace_printer.h"

namespace __sanitizer {

static const char *StripFunctionName(const char *function, const char *prefix) {
  if (!function) return nullptr;
  if (!prefix) return function;
  uptr prefix_len = internal_strlen(prefix);
  if (0 == internal_strncmp(function, prefix, prefix_len))
    return function + prefix_len;
  return function;
}

static const char kDefaultFormat[] = "    #%n %p %F %L";

void RenderFrame(InternalScopedString *buffer, const char *format, int frame_no,
                 const AddressInfo &info, bool vs_style,
                 const char *strip_path_prefix, const char *strip_func_prefix) {
  if (0 == internal_strcmp(format, "DEFAULT"))
    format = kDefaultFormat;
  for (const char *p = format; *p != '\0'; p++) {
    if (*p != '%') {
      buffer->append("%c", *p);
      continue;
    }
    p++;
    switch (*p) {
    case '%':
      buffer->append("%%");
      break;
    // Frame number and all fields of AddressInfo structure.
    case 'n':
      buffer->append("%zu", frame_no);
      break;
    case 'p':
      buffer->append("0x%zx", info.address);
      break;
    case 'm':
      buffer->append("%s", StripPathPrefix(info.module, strip_path_prefix));
      break;
    case 'o':
      buffer->append("0x%zx", info.module_offset);
      break;
    case 'f':
      buffer->append("%s", StripFunctionName(info.function, strip_func_prefix));
      break;
    case 'q':
      buffer->append("0x%zx", info.function_offset != AddressInfo::kUnknown
                                  ? info.function_offset
                                  : 0x0);
      break;
    case 's':
      buffer->append("%s", StripPathPrefix(info.file, strip_path_prefix));
      break;
    case 'l':
      buffer->append("%d", info.line);
      break;
    case 'c':
      buffer->append("%d", info.column);
      break;
    // Smarter special cases.
    case 'F':
      // Function name and offset, if file is unknown.
      if (info.function) {
        buffer->append("in %s",
                       StripFunctionName(info.function, strip_func_prefix));
        if (!info.file && info.function_offset != AddressInfo::kUnknown)
          buffer->append("+0x%zx", info.function_offset);
      }
      break;
    case 'S':
      // File/line information.
      RenderSourceLocation(buffer, info.file, info.line, info.column, vs_style,
                           strip_path_prefix);
      break;
    case 'L':
      // Source location, or module location.
      if (info.file) {
        RenderSourceLocation(buffer, info.file, info.line, info.column,
                             vs_style, strip_path_prefix);
      } else if (info.module) {
        RenderModuleLocation(buffer, info.module, info.module_offset,
                             strip_path_prefix);
      } else {
        buffer->append("(<unknown module>)");
      }
      break;
    case 'M':
      // Module basename and offset, or PC.
      if (info.address & kExternalPCBit)
        {} // There PCs are not meaningful.
      else if (info.module)
        buffer->append("(%s+%p)", StripModuleName(info.module),
                       (void *)info.module_offset);
      else
        buffer->append("(%p)", (void *)info.address);
      break;
    default:
      Report("Unsupported specifier in stack frame format: %c (0x%zx)!\n", *p,
             *p);
      Die();
    }
  }
}

void RenderData(InternalScopedString *buffer, const char *format,
                const DataInfo *DI, const char *strip_path_prefix) {
  for (const char *p = format; *p != '\0'; p++) {
    if (*p != '%') {
      buffer->append("%c", *p);
      continue;
    }
    p++;
    switch (*p) {
      case '%':
        buffer->append("%%");
        break;
      case 's':
        buffer->append("%s", StripPathPrefix(DI->file, strip_path_prefix));
        break;
      case 'l':
        buffer->append("%d", DI->line);
        break;
      case 'g':
        buffer->append("%s", DI->name);
        break;
      default:
        Report("Unsupported specifier in stack frame format: %c (0x%zx)!\n", *p,
               *p);
        Die();
    }
  }
}

void RenderSourceLocation(InternalScopedString *buffer, const char *file,
                          int line, int column, bool vs_style,
                          const char *strip_path_prefix) {
  if (vs_style && line > 0) {
    buffer->append("%s(%d", StripPathPrefix(file, strip_path_prefix), line);
    if (column > 0)
      buffer->append(",%d", column);
    buffer->append(")");
    return;
  }

  buffer->append("%s", StripPathPrefix(file, strip_path_prefix));
  if (line > 0) {
    buffer->append(":%d", line);
    if (column > 0)
      buffer->append(":%d", column);
  }
}

void RenderModuleLocation(InternalScopedString *buffer, const char *module,
                          uptr offset, const char *strip_path_prefix) {
  buffer->append("(%s+0x%zx)", StripPathPrefix(module, strip_path_prefix),
                 offset);
}

} // namespace __sanitizer

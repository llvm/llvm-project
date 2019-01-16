#include "xpc/Conversion.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <dlfcn.h>
#include <stdio.h>
#include <string>
#include <xpc/xpc.h>

typedef const char *(*clangd_xpc_get_bundle_identifier_t)(void);

using namespace llvm;
using namespace clang;

std::string getLibraryPath() {
  Dl_info info;
  if (dladdr((void *)(uintptr_t)getLibraryPath, &info) == 0)
    llvm_unreachable("Call to dladdr() failed");
  llvm::SmallString<128> LibClangPath;
  LibClangPath = llvm::sys::path::parent_path(
      llvm::sys::path::parent_path(info.dli_fname));
  llvm::sys::path::append(LibClangPath, "lib", "ClangdXPC.framework",
                          "ClangdXPC");
  return LibClangPath.str();
}

static void dumpXPCObject(xpc_object_t Object, llvm::raw_ostream &OS) {
  xpc_type_t Type = xpc_get_type(Object);
  if (Type == XPC_TYPE_DICTIONARY) {
    json::Value Json = clang::clangd::xpcToJson(Object);
    OS << Json;
  } else {
    OS << "<UNKNOWN>";
  }
}

int main(int argc, char *argv[]) {
  // Open the ClangdXPC dylib in the framework.
  std::string LibPath = getLibraryPath();
  void *dlHandle = dlopen(LibPath.c_str(), RTLD_LOCAL | RTLD_FIRST);
  if (!dlHandle)
    return 1;

  // Lookup the XPC service bundle name, and launch it.
  clangd_xpc_get_bundle_identifier_t clangd_xpc_get_bundle_identifier =
      (clangd_xpc_get_bundle_identifier_t)dlsym(
          dlHandle, "clangd_xpc_get_bundle_identifier");
  xpc_connection_t conn = xpc_connection_create(
      clangd_xpc_get_bundle_identifier(), dispatch_get_main_queue());

  // Dump the XPC events.
  xpc_connection_set_event_handler(conn, ^(xpc_object_t event) {
    if (event == XPC_ERROR_CONNECTION_INVALID) {
      llvm::errs() << "Received XPC_ERROR_CONNECTION_INVALID.";
      exit(EXIT_SUCCESS);
    }
    if (event == XPC_ERROR_CONNECTION_INTERRUPTED) {
      llvm::errs() << "Received XPC_ERROR_CONNECTION_INTERRUPTED.";
      exit(EXIT_SUCCESS);
    }

    dumpXPCObject(event, llvm::outs());
    llvm::outs() << "\n";
  });

  xpc_connection_resume(conn);

  // Read the input to determine the things to send to clangd.
  llvm::ErrorOr<std::unique_ptr<MemoryBuffer>> Stdin =
      llvm::MemoryBuffer::getSTDIN();
  if (!Stdin) {
    llvm::errs() << "Failed to get STDIN!\n";
    return 1;
  }
  for (llvm::line_iterator It(**Stdin, /*SkipBlanks=*/true,
                              /*CommentMarker=*/'#');
       !It.is_at_eof(); ++It) {
    StringRef Line = *It;
    if (auto Request = json::parse(Line)) {
      xpc_object_t Object = clangd::jsonToXpc(*Request);
      xpc_connection_send_message(conn, Object);
    } else {
      llvm::errs() << llvm::Twine("JSON parse error: ")
                   << llvm::toString(Request.takeError());
      return 1;
    }
  }

  dispatch_main();

  // dispatch_main() doesn't return
  return EXIT_FAILURE;
}

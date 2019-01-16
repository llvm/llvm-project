
/// Returns the bundle identifier of the Clangd XPC service.
extern "C" const char *clangd_xpc_get_bundle_identifier() {
  return "org.llvm.clangd";
}

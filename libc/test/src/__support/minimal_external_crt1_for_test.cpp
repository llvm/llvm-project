// A minimal external crt1 object used only by link tests.
// Executables linked with this object must not be run.

extern "C" [[noreturn]] void _start() {
  for (;;) {
  }
}

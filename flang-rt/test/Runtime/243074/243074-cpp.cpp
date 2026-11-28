// 243074-cpp.cpp: C++ companion file to test defined in 243074-f90.f90.
extern "C" void report_at_exit();
namespace {
struct Reporter {
  ~Reporter() { report_at_exit(); }
} reporter;
} // namespace

// RUN: %check_clang_tidy %s pybind-call-guard-init %t

namespace pybind11 {

namespace detail {
template <typename... Args>
struct constructor {};

template <typename Func>
struct factory {};
} // namespace detail

template <typename... Args>
detail::constructor<Args...> init() {
  return detail::constructor<Args...>();
}

template <typename Func>
detail::factory<Func> init(Func &&) {
  return detail::factory<Func>();
}

struct gil_scoped_release {};

template <typename... Ts>
struct call_guard {};

struct arg {
  arg(const char *) {}
};

template <typename Class>
struct class_ {
  template <typename... Args>
  class_ &def(Args &&...);
};

} // namespace pybind11

namespace py = pybind11;

struct CustomGuard {};

struct Foo {
  Foo();
  Foo(int, int);
  Foo(int);
  void bar();
};

void register_foo(py::class_<Foo> &cl) {
  cl.def(py::init<>(), py::call_guard<py::gil_scoped_release>());
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: do not use 'py::call_guard<py::gil_scoped_release>' on 'py::init'; release the GIL inside the factory function body instead [pybind-call-guard-init]

  cl.def(py::init<int, int>(), py::arg("a"), py::arg("b"), py::call_guard<py::gil_scoped_release>());
  // CHECK-MESSAGES: :[[@LINE-1]]:60: warning: do not use 'py::call_guard<py::gil_scoped_release>' on 'py::init'; release the GIL inside the factory function body instead [pybind-call-guard-init]

  cl.def(py::init([](int x) { return new Foo(x); }), py::call_guard<py::gil_scoped_release>());
  // CHECK-MESSAGES: :[[@LINE-1]]:54: warning: do not use 'py::call_guard<py::gil_scoped_release>' on 'py::init'; release the GIL inside the factory function body instead [pybind-call-guard-init]

  cl.def(py::init([](int x) { return new Foo(x); }), py::call_guard<CustomGuard, py::gil_scoped_release>());
  // CHECK-MESSAGES: :[[@LINE-1]]:54: warning: do not use 'py::call_guard<py::gil_scoped_release>' on 'py::init'; release the GIL inside the factory function body instead [pybind-call-guard-init]

  // Valid usages should not warn
  cl.def("bar", &Foo::bar, py::call_guard<py::gil_scoped_release>());
  cl.def(py::init<>());
  cl.def(py::init<int, int>(), py::arg("a"), py::arg("b"));
  cl.def(py::init<int>(), py::call_guard<CustomGuard>());
  cl.def(py::init([](int x) {
    py::gil_scoped_release nogil;
    return new Foo(x);
  }));
}

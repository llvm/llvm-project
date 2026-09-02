```{title} clang-tidy - pybind-call-guard-init
```

# pybind-call-guard-init

Finds instances where `py::call_guard<py::gil_scoped_release>()` is passed
to `py::class_::def(...)` alongside `py::init(...)`.

Using `py::call_guard<py::gil_scoped_release>()` on `py::init(...)` keeps the
Python Global Interpreter Lock (GIL) released during constructor trampoline
execution, which causes pybind11's internal Python object initialization and
instance registration to run without holding the GIL.

Instead, release the GIL only inside the factory function or lambda body:

```cpp
// Incorrect:
cl.def(py::init([](int arg) {
         return std::make_unique<MyClass>(arg);
       }),
       py::call_guard<py::gil_scoped_release>());

// Correct:
cl.def(py::init([](int arg) {
         py::gil_scoped_release nogil;
         return std::make_unique<MyClass>(arg);
       }));
```

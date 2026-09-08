// RUN: %clang_cc1 -std=c++26 -fherbceptions -fcxx-exceptions -fexceptions -fsyntax-only -verify %s
//
// Template, if constexpr, and OOP herbception tests. `throw throws` inside a
// template must defer error_domain lookup to instantiation time; the
// fabrication must succeed for any concrete type T that has an
// error_domain<T> specialization, and fail for types that do not.

namespace std {

struct error_domain_singleton {
  void (*do_cleanup)(unsigned long) noexcept = 0;
  bool (*do_equivalent)(unsigned long, error_domain_singleton const *,
                        unsigned long) noexcept = 0;
  void (*do_name)(unsigned long, int, void *, void *) noexcept = 0;
  void (*do_message)(unsigned long, int, void *, void *) noexcept = 0;
  int (*do_to_errc)(unsigned long) noexcept = 0;
};

class error {
public:
  error() = delete;
  error(error const &) = delete;
  error &operator=(error const &) = delete;
  constexpr ~error() noexcept {}
  __SIZE_TYPE__ code() const noexcept { return code_opaque; }

private:
  void const *domain_opaque{};
  __SIZE_TYPE__ code_opaque{};
  explicit constexpr error(void const *domain, __SIZE_TYPE__ code) noexcept
      : domain_opaque(domain), code_opaque(code) {}
};

template <typename T> struct error_domain;

enum class errc : int {
  invalid_argument = 22,
  bad_address = 14,
  io_error = 5,
};

const error_domain_singleton dummy_domain{};

template <> struct error_domain<errc> {
  static constexpr error_domain_singleton const *domain() noexcept {
    return &dummy_domain;
  }
  static constexpr __SIZE_TYPE__ code(errc e) noexcept {
    return static_cast<unsigned long>(e);
  }
};

} // namespace std

// --- 1. Basic template: throw throws with dependent operand type ---

template <typename T>
void throw_template(T ec) throws {
  throw throws ec; // ok: fabrication deferred to instantiation
}

void use_throw_template() {
  throw_template(::std::errc::io_error); // ok: errc has error_domain
}

// --- 2. Template instantiated with type lacking error_domain: hard error ---
// The error fires at the template body, with a note at the call site.

struct no_domain_type {};

template <typename T>
void throw_template_no_domain(T ec) throws {
  throw throws ec; // expected-error {{'throw throws' operand type 'no_domain_type' has no std::error_domain specialization}}
}

void use_throw_template_no_domain() {
  throw_template_no_domain(no_domain_type{}); // expected-note {{in instantiation of function template specialization 'throw_template_no_domain<no_domain_type>' requested here}}
}

// --- 3. if constexpr: branch discarded for types without domain ---

template <typename T>
void if_constexpr_throw(T ec) throws {
  if constexpr (requires {
    typename std::error_domain<T>;
    std::error_domain<T>::domain();
    std::error_domain<T>::code(ec);
  }) {
    throw throws ec;
  }
}

void use_if_constexpr_throw() {
  if_constexpr_throw(::std::errc::io_error);  // ok: branch live, errc has domain
  if_constexpr_throw(no_domain_type{});        // ok: branch discarded at instantiation
}

// --- 4. OOP: throw throws inside a member function template ---

struct Widget {
  template <typename T>
  void process(T ec) throws {
    throw throws ec;
  }

  template <typename T>
  void method_try_block(T ec)
  try {
    throw throws ec;
  } catch throws(::std::error e) {
    (void)e.code();
  }
};

void use_widget() {
  Widget w;
  w.process(::std::errc::bad_address);
}

// --- 5. OOP: virtual throws function ---

struct Base {
  virtual void do_throw(::std::errc ec) throws {
    throw throws ec;
  }
  virtual ~Base() = default;
};

struct Derived : Base {
  void do_throw(::std::errc ec) throws {
    if (ec == ::std::errc::io_error)
      throw throws ec;
  }
};

void use_derived() {
  Derived d;
  d.do_throw(::std::errc::io_error);
  Base *b = &d;
  b->do_throw(::std::errc::bad_address);
}

// --- 6. Template with function-try-block ---

template <typename T>
void fn_try_block(T ec)
try {
  throw throws ec;
} catch throws(::std::error e) {
  (void)e.code();
}

void use_fn_try_block() {
  fn_try_block(::std::errc::invalid_argument);
}

// --- 7. Template with catch throws and rethrow ---

template <typename T>
void catch_and_reref(T ec) throws {
  try {
    throw throws ec;
  } catch throws(::std::error e) {
    if (e.code() == static_cast<__SIZE_TYPE__>(ec))
      throw throws; // bare rethrow
    throw throws ::std::errc::bad_address;
  }
}

void use_catch_and_reref() {
  catch_and_reref(::std::errc::io_error);
}

// --- 8. Constexpr template throw ---

template <typename T>
constexpr int constexpr_throw(T ec) throws {
  if (static_cast<int>(ec) == 0)
    throw throws ec;
  return static_cast<int>(ec);
}

template <typename T>
constexpr int constexpr_try(T ec) {
  try {
    return constexpr_throw(ec);
  } catch throws(::std::error e) {
    return static_cast<int>(e.code());
  }
}

static_assert(constexpr_try<::std::errc>(::std::errc::invalid_argument) == 22, "");
static_assert(constexpr_try<::std::errc>(::std::errc::io_error) == 5, "");

// --- 9. Template with explicit specialization for no_domain_type ---

template <typename T>
void specialized_throw(T ec) throws {
  throw throws ec;
}

// Explicit specialization: no_domain_type gets a no-op (no throw).
template <>
void specialized_throw<no_domain_type>(no_domain_type) throws {}

void use_specialized() {
  specialized_throw(::std::errc::bad_address);
  specialized_throw(no_domain_type{}); // ok: uses the specialization
}

// --- 10. Member function template with type lacking error_domain ---

struct Processor {
  template <typename T>
  void handle(T ec) throws {
    throw throws ec; // expected-error {{'throw throws' operand type 'no_domain_type' has no std::error_domain specialization}}
  }
};

void use_processor_no_domain() {
  Processor p;
  p.handle(no_domain_type{}); // expected-note {{in instantiation of function template specialization 'Processor::handle<no_domain_type>' requested here}}
}

// --- 11. if constexpr with throw throws inside a non-dependent context ---

void non_template_if_constexpr(int x) throws {
  if constexpr (sizeof(int) > 2) {
    throw throws ::std::errc::io_error; // ok: branch always live, errc has domain
  }
}

// --- 12. Nested templates ---

template <typename T>
struct Outer {
  template <typename U>
  void nested(T a, U b) throws {
    if (static_cast<int>(a) == 0)
      throw throws b;
  }
};

void use_nested() {
  Outer<::std::errc> o;
  o.nested(::std::errc::bad_address, ::std::errc::io_error); // ok
}

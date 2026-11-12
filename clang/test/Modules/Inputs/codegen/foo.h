inline void f1(const char* fmt, ...) {
  __builtin_va_list args;
  __builtin_va_start(args, fmt);
}

struct non_trivial_dtor {
  ~non_trivial_dtor();
};

struct implicit_dtor {
  non_trivial_dtor d;
};

struct uninst_implicit_dtor {
  non_trivial_dtor d;
};

inline void use_implicit_dtor() {
  implicit_dtor d;
}

template <typename T>
void inst() {
}

inline void inst_decl() {
  // cause inst<int>'s declaration to be instantiated, without a definition.
  (void)sizeof(&inst<int>);
  inst<float>();
}

__attribute__((always_inline)) inline void always_inl() {
}

asm("narf");

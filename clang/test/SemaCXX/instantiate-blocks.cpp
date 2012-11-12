// RUN: %clang_cc1 -fblocks -fsyntax-only -verify %s
// rdar: // 6182276

template <typename T, typename T1> void foo(T t, T1 r)
{
    T block_arg;
    __block T1 byref_block_arg;

    T1 (^block)(T)  =  ^ T1 (T arg) { 
         byref_block_arg = arg;
         block_arg = arg; 	// expected-error {{variable is not assignable (missing __block type specifier)}}
         return block_arg+arg; };
}

// rdar://10466373
template <typename T, typename T1> void noret(T t, T1 r)
{
    (void) ^{
    if (1)
      return t;
    else if (2)
      return r;  // expected-error {{return type 'double' must match previous return type 'float' when block literal has unspecified explicit return type}}
  };
}

int main(void)
{
    foo(100, 'a');	// expected-note {{in instantiation of function template specialization 'foo<int, char>' requested here}}

   noret((float)0.0, double(0.0)); // expected-note {{in instantiation of function template specialization 'noret<float, double>' requested here}}
}


// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -finclude-default-header -fsyntax-only -verify %s

Texture2D<float4> tex;

void main() {
  uint u_w, u_h, u_l;
  float f_w, f_h, f_l;

  // Valid calls
  tex.GetDimensions(u_w, u_h);
  tex.GetDimensions(0, u_w, u_h, u_l);
  tex.GetDimensions(f_w, f_h);
  tex.GetDimensions(0, f_w, f_h, f_l);

  // Invalid number of arguments
  // expected-error@+1 {{no matching member function for call to 'GetDimensions'}}
  tex.GetDimensions();
  // expected-note@*:* 2 {{candidate function not viable: requires 2 arguments, but 0 were provided}}
  // expected-note@*:* 2 {{candidate function not viable: requires 4 arguments, but 0 were provided}}

  // expected-error@+1 {{no matching member function for call to 'GetDimensions'}}
  tex.GetDimensions(u_w);
  // expected-note@*:* 2 {{candidate function not viable: requires 2 arguments, but 1 was provided}}
  // expected-note@*:* 2 {{candidate function not viable: requires 4 arguments, but 1 was provided}}

  // expected-error@+1 {{no matching member function for call to 'GetDimensions'}}
  tex.GetDimensions(u_w, u_h, u_l);
  // expected-note@*:* 2 {{candidate function not viable: requires 2 arguments, but 3 were provided}}
  // expected-note@*:* 2 {{candidate function not viable: requires 4 arguments, but 3 were provided}}

  // expected-error@+1 {{no matching member function for call to 'GetDimensions'}}
  tex.GetDimensions(0, u_w, u_h, u_l, 0);
  // expected-note@*:* 2 {{candidate function not viable: requires 4 arguments, but 5 were provided}}
  // expected-note@*:* 2 {{candidate function not viable: requires 2 arguments, but 5 were provided}}

  // Invalid types
  int i_w, i_h;
  // expected-error@+1 {{no matching member function for call to 'GetDimensions'}}
  tex.GetDimensions(i_w, i_h);
  // expected-note@*:* {{candidate function not viable: no known conversion from 'int' to 'unsigned int &__restrict' for 1st argument}}
  // expected-note@*:* {{candidate function not viable: no known conversion from 'int' to 'float &__restrict' for 1st argument}}
  // expected-note@*:* 2 {{candidate function not viable: requires 4 arguments, but 2 were provided}}

  // expected-error@+1 {{no matching member function for call to 'GetDimensions'}}
  tex.GetDimensions(u_w, i_h);
  // expected-note@*:* {{candidate function not viable: no known conversion from 'int' to 'unsigned int &__restrict' for 2nd argument}}
  // expected-note@*:* {{candidate function not viable: no known conversion from 'uint' (aka 'unsigned int') to 'float &__restrict' for 1st argument}}
  // expected-note@*:* 2 {{candidate function not viable: requires 4 arguments, but 2 were provided}}

  // Invalid lvalues
  // expected-error@+1 {{no matching member function for call to 'GetDimensions'}}
  tex.GetDimensions(0u, u_h);
  // expected-note@*:* {{candidate function not viable: expects an lvalue for 1st argument}}
  // expected-note@*:* {{candidate function not viable: no known conversion from 'unsigned int' to 'float &__restrict' for 1st argument}}
  // expected-note@*:* 2 {{candidate function not viable: requires 4 arguments, but 2 were provided}}

  // expected-error@+1 {{no matching member function for call to 'GetDimensions'}}
  tex.GetDimensions(u_w, 0u);
  // expected-note@*:* {{candidate function not viable: expects an lvalue for 2nd argument}}
  // expected-note@*:* {{candidate function not viable: no known conversion from 'uint' (aka 'unsigned int') to 'float &__restrict' for 1st argument}}
  // expected-note@*:* 2 {{candidate function not viable: requires 4 arguments, but 2 were provided}}

  // expected-error@+1 {{no matching member function for call to 'GetDimensions'}}
  tex.GetDimensions(0, 0u, u_h, u_l);
  // expected-note@*:* {{candidate function not viable: expects an lvalue for 2nd argument}}
  // expected-note@*:* {{candidate function not viable: no known conversion from 'unsigned int' to 'float &__restrict' for 2nd argument}}
  // expected-note@*:* 2 {{candidate function not viable: requires 2 arguments, but 4 were provided}}

  // expected-error@+1 {{no matching member function for call to 'GetDimensions'}}
  tex.GetDimensions(0, u_w, 0u, u_l);
  // expected-note@*:* {{candidate function not viable: expects an lvalue for 3rd argument}}
  // expected-note@*:* {{candidate function not viable: no known conversion from 'uint' (aka 'unsigned int') to 'float &__restrict' for 2nd argument}}
  // expected-note@*:* 2 {{candidate function not viable: requires 2 arguments, but 4 were provided}}

  // expected-error@+1 {{no matching member function for call to 'GetDimensions'}}
  tex.GetDimensions(0, u_w, u_h, 0u);
  // expected-note@*:* {{candidate function not viable: expects an lvalue for 4th argument}}
  // expected-note@*:* {{candidate function not viable: no known conversion from 'uint' (aka 'unsigned int') to 'float &__restrict' for 2nd argument}}
  // expected-note@*:* 2 {{candidate function not viable: requires 2 arguments, but 4 were provided}}
}

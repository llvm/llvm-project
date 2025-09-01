// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.6-library %s -fnative-half-type -emit-llvm-only -disable-llvm-passes -verify

export double2 double_vec_mod_by_int(double2 p1) {
    return  p1 % 2;
    // expected-error@-1 {{invalid operands to binary expression ('double2' (aka 'vector<double, 2>') and 'int')}}
}

export double2 double_vec_mod_by_float(double2 p1) {
    return  p1 % 2.0;
    // expected-error@-1 {{invalid operands to binary expression ('double2' (aka 'vector<double, 2>') and 'float')}}
}

export double2 double_vec_mod_by_double(double2 p1, double p2 ) {
    return  p1 % p2;
    // expected-error@-1 {{invalid operands to binary expression ('double2' (aka 'vector<double, 2>') and 'double')}}
}

export double2 double_vec_mod_by_double_vec(double2 p1, double2 p2 ) {
    return  p1 % p2;
    // expected-error@-1 {{invalid operands to binary expression ('double2' (aka 'vector<double, 2>') and 'double2')}}
}

export double double_mod_by_int(double p1) {
    return  p1 % 2;
    // expected-error@-1 {{invalid operands to binary expression ('double' and 'int')}}
}

export double double_mod_by_float(double p1) {
    return  p1 % 2.0;
    // expected-error@-1 {{invalid operands to binary expression ('double' and 'float')}}
}

export double double_mod_by_double(double p1, double p2 ) {
    return  p1 % p2;
    // expected-error@-1 {{invalid operands to binary expression ('double' and 'double')}}
}

export double double_mod_by_double_vec(double p1, double2 p2 ) {
    return  p1 % p2;
    // expected-error@-1 {{invalid operands to binary expression ('double' and 'double2' (aka 'vector<double, 2>'))}}
}

// For backward compatibility, fields of C unions declared in system headers
// that have non-trivial ObjC ownership qualifications are marked as unavailable
// unless the qualifier is explicit and __strong.

#pragma clang system_header

typedef __strong id StrongID;

typedef union {
  id f0;
  _Nonnull id f1;
  __weak id f2;
  StrongID f3;
} U0_SystemHeader;

typedef union { // expected-note {{'U1_SystemHeader' has subobjects that are non-trivial to destruct}} expected-note {{'U1_SystemHeader' has subobjects that are non-trivial to copy}}
  __strong id f0; // expected-note {{f0 has type '__strong id' that is non-trivial to destruct}} expected-note {{f0 has type '__strong id' that is non-trivial to copy}}
  _Nonnull id f1;
} U1_SystemHeader;

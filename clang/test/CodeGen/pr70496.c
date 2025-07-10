// RUN: %clang --target=wasm32 -mmultivalue -Xclang -target-abi -Xclang experimental-mv %s -S -Xclang -verify

float crealf() { return 0;} // expected-no-diagnostics
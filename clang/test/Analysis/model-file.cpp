// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-config faux-bodies=true,model-path=%S/Inputs/Models -analyzer-output=plist-multi-file -verify %s -o %t
// RUN: cat %t | %diff_plist %S/Inputs/expected-plists/model-file.cpp.plist -

typedef int* intptr;

// This function is modeled and the p pointer is dereferenced in the model
// function and there is no function definition available. The modeled
// function can use any types that are available in the original translation
// unit, for example intptr in this case.
void modeledFunction(intptr p);

// This function is modeled and returns true if the parameter is not zero
// and there is no function definition available.
bool notzero(int i);

// This functions is not modeled and there is no function definition.
// available
bool notzero_notmodeled(int i);

int main() {
  // There is a nullpointer dereference inside this function.
  modeledFunction(0);

  int p = 0;
  if (notzero(p)) {
   // It is known that p != 0 because of the information provided by the
   // model of the notzero function.
    int j = 5 / p;
  }

  if (notzero_notmodeled(p)) {
   // There is no information about the value of p, because
   // notzero_notmodeled is not modeled and the function definition
   // is not available.
    int j = 5 / p; // expected-warning {{Division by zero}}
  }

  return 0;
}


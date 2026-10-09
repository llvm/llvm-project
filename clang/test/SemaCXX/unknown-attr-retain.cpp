// Retaining an unknown [[...]] attribute must not add any diagnostic beyond the
// existing -Wunknown-attributes warning. In particular an unknown attribute on a
// statement must NOT be reported as "invalid on a statement": an unrecognized
// attribute-token is ignored, [dcl.attr.grammar]/8
// (https://eel.is/c++draft/dcl.attr.grammar#8), not misapplied. Checked across
// standard modes, since attribute appertainment is language-version sensitive.

// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

// On a declaration.
struct X {
  int x [[ns::transient(a, b)]]; // expected-warning {{unknown attribute 'ns::transient' ignored}}
};

void f() {
  // On a statement: only the unknown-attribute warning, no appertainment error.
  [[ns::transient(a, b)]] { // expected-warning {{unknown attribute 'ns::transient' ignored}}
  }

  // Unknown attribute with no argument clause.
  [[frobble]] while (false) {} // expected-warning {{unknown attribute 'frobble' ignored}}
}

// Negative case: a recognized attribute is unaffected by retention -- it is
// applied normally and produces no unknown-attribute warning (and so is not
// turned into an UnknownAttr).
[[nodiscard]] int g();

void neg() {
  // Consistency at the boundary: a recognized attribute misapplied to a
  // statement still gets its normal diagnostic and is not retained. Only
  // genuinely unknown attributes take the retention path, on statements just as
  // on declarations.
  [[fallthrough]]; // expected-error {{fallthrough annotation is outside switch statement}}
}

// Same boundary in type position: a recognized type attribute that fails its
// own validation still diagnoses normally and is not retained as an
// UnknownTypeAttr. Retention is limited to genuinely unknown attributes across
// declaration, statement, and type positions alike.
typedef int BadVec [[gnu::vector_size(3)]]; // expected-error {{vector size not an integral multiple of component size}}

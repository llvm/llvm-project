} // expected-error {{extraneous closing brace ('}')}}
int not_in_extern_c;
extern "C" { // expected-note {{to match this '{'}}
// expected-error {{expected '}'}}

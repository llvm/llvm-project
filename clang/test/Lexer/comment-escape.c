// RUN: %clang -fsyntax-only -Wdocumentation %s
// rdar://6757323
// foo \

#define blork 32

// GH62054

/**<*\
/
//expected-warning@-2 {{escaped newline between}} \
//expected-warning@-2 {{line splicing in Doxygen comments are not supported}}

/**<*\	
/
//expected-warning@-2 {{escaped newline between}} \
//expected-warning@-2 {{backslash and newline separated by space}} \
//expected-warning@-2 {{line splicing in Doxygen comments are not supported}}


/*<*\
/
//expected-warning@-2 {{escaped newline between}}  \
//expected-warning@-2 {{line splicing in Doxygen comments are not supported}}

/*<*\	
/
//expected-warning@-2 {{escaped newline between}} \
//expected-warning@-2 {{backslash and newline separated by space}} \
//expected-warning@-2 {{line splicing in Doxygen comments are not supported}}

/\
*<**/
//expected-warning@-2 {{line splicing in Doxygen comments are not supported}}

/\
/<*
//expected-warning@-2 {{line splicing in Doxygen comments are not supported}}

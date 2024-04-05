
struct mystruct {
  long a;
  long b;
  long c;
};
void noncompliant_f1(void);
void compliant_f1(void);
void noncompliant_f3(struct mystruct *msptr); 
void compliant_f3(struct mystruct *msptr);
extern void sink(const char *);

enum { bufsize = 1024 };

void noncompliant_f1(void) {
  int buffer[bufsize]; 

  int *bptr = &buffer[0]; 
  int *ptr  = bptr; 
  while ( ptr < bptr + sizeof(buffer) ) { // noncompliant
    *ptr++ = 0;	// compliant  
  }
}
void noncompliant_f1a(void) {
  typedef int my_int_t;	
  my_int_t buffer[bufsize]; 

  my_int_t *bptr = &buffer[0]; 
  my_int_t *ptr  = bptr; 
  while ( ptr < bptr + sizeof(buffer) ) { // noncompliant
    *ptr++ = 0;	// compliant  
  }
}
void compliant_f1(void) {
  int buffer[bufsize]; 

  int *bptr = &buffer[0]; 
  int *ptr  = bptr; 
  while ( ptr < bptr + bufsize ) { // compliant
    *ptr++ = 0;	// compliant  
  }
}

void noncompliant_f2(void) {
  int buffer[bufsize];
  int *ptr = buffer;

  while ( ptr < buffer + sizeof(buffer) ) { // noncompliant
    *ptr++ = 0;	// compliant
  }
}

void compliant_f2(void) {
  int buffer[bufsize];
  int *ptr  = buffer; 

  while ( ptr < buffer + bufsize ) { // compliant
    *ptr++ = 0;	// compliant  
  }
}

void memset2(void*, int, unsigned int);

void noncompliant_f3(struct mystruct *msptr) {
  const unsigned int skip = sizeof(long); // why offsetof is declared?
  struct mystruct *ptr = msptr;
  
  memset2(ptr + skip, // noncompliant, impossible with tidy
                     0, sizeof(struct mystruct) - skip);
}

void compliant_f3(struct mystruct *msptr) {
  const unsigned int skip = sizeof(long);
  char *ptr = (char*)msptr;

  memset2(ptr + skip, // compliant
                     0, sizeof(struct mystruct) - skip); 
}

void noncompliant_f4(void) {
  int buffer[bufsize]; 

  int *bptr = &buffer[0]; 
  int *ptr  = bptr; 
  while ( ptr < bptr + bufsize ) { // compliant
    *ptr = 0;	  
    ptr += sizeof(*ptr); // noncompliant
  }
}
void noncompliant_f4w(void) { /* accidentally good */
  char buffer[bufsize]; 

  char *bptr = &buffer[0]; 
  char *ptr  = bptr; 
  while ( ptr < bptr + bufsize ) { // compliant
    *ptr = 0;	  
    ptr += sizeof(*ptr);  // silenced
  }
}
void compliant_f4w(void) {
  char buffer[bufsize]; 

  char *bptr = &buffer[0]; 
  char *ptr  = bptr; 
  while ( ptr < bptr + bufsize ) { // compliant
    *ptr = 0;	  
    ptr += 1;  // compliant
  }
}

void noncompliant_f5(void) {
  int buffer[bufsize]; 

  int *bptr = &buffer[0]; 
  int *ptr  = bptr; 
  while ( ptr < bptr + bufsize ) { // compliant
    *ptr = 0;	  
    ptr = ptr + sizeof(*ptr); // noncompliant
  }
}
void noncompliant_f5w(void) {
  char buffer[bufsize]; 

  char*bptr = &buffer[0]; 
  char *ptr  = bptr; 
  while ( ptr < bptr + bufsize ) { // compliant
    *ptr = 0;	  
    ptr = ptr + sizeof(*ptr); // silenced
  }
}
void noncompliant_f5c(void) {
  char buffer[bufsize]; 

  char*bptr = &buffer[0]; 
  const char *ptr  = bptr; 
  while ( ptr < bptr + bufsize ) { // compliant
    sink(ptr);	  
    ptr = ptr + sizeof(*ptr); // silenced
  }
}
void compliant_f5c(void) {
  char buffer[bufsize]; 

  char *bptr = &buffer[0]; 
  const char *ptr  = bptr; 
  while ( ptr < bptr + bufsize ) { // compliant
    sink(ptr);	  
    ptr = ptr + 1;  // compliant
  }
}

void noncompliant_f6(void) {
  int buffer[bufsize]; 

  int *bptr = &buffer[0]; 
  int *ptr  = bptr + bufsize; // compliant
  while ( ptr >= bptr ) {
    *ptr = 0;	  
    ptr = ptr - sizeof(*ptr); // noncompliant
  }
}
void noncompliant_f6w(void) {
  char buffer[bufsize]; 

  char *bptr = &buffer[0]; 
  char *ptr  = bptr + bufsize; // compliant
  while ( ptr >= bptr ) {
    *ptr = 0;	  
    ptr = ptr - sizeof(*ptr); // silenced
  }
}
void compliant_f6(void) {
  int buffer[bufsize]; 

  int *bptr = &buffer[0]; 
  int *ptr  = bptr + bufsize; // compliant
  while ( ptr >= bptr ) {
    *ptr = 0;	  
    ptr = ptr - 1; // compliant
  }
}

void compliant_f7(void) {
  int buffer[bufsize]; 

  int *bptr = &buffer[0]; 
  int *ptr  = bptr + bufsize; // compliant
  int i = ptr - bptr; // compliant
  while ( i >= 0 ) {
    ptr[i] = 0;	  
    i = i - 1; // compliant
  }
}

void compliant_f8(void) {
  int buffer[bufsize]; 

  int *bptr = &buffer[0]; 
  int *ptr  = bptr + bufsize; // compliant
  int i = sizeof(*ptr) - sizeof(*bptr); // compliant
}
void compliant_f9(void) {
  int buffer[bufsize]; 

  int *bptr = &buffer[0]; 
  int *ptr  = bptr + bufsize; // compliant
  int i = sizeof(ptr) - sizeof(*bptr); // compliant
}


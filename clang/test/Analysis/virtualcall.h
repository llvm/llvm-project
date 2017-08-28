namespace header {
  class Z {
  public:
    Z() {
      foo();
#if !PUREONLY
	// expected-warning-re@-2 {{{{^}}Call to virtual function during construction}}
	// expected-note-re@-3 {{{{^}}This constructor of an object of type 'Z' has not returned when the virtual method was called}}
	// expected-note-re@-4 {{{{^}}Call to virtual function during construction}}	
#endif
    }
    virtual int foo();
  };
}

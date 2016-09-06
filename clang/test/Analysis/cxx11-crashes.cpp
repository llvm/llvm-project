// RUN: %clang_cc1 -analyze -analyzer-checker=core -std=c++11 -verify %s

// radar://11485149, PR12871
class PlotPoint {
  bool valid;
};

PlotPoint limitedFit () {
  PlotPoint fit0;
  fit0 = limitedFit ();
  return fit0;
}

// radar://11487541, NamespaceAlias
namespace boost {namespace filesystem3 {
class path {
public:
 path(){}
};

}}
namespace boost
{
  namespace filesystem
  {
    using filesystem3::path;
  }
}

void radar11487541() {
  namespace fs = boost::filesystem;
  fs::path p;
}

// PR12873 radar://11499139
void testFloatInitializer() {
  const float ysize={0.015}, xsize={0.01};
}


// PR12874, radar://11487525
template<class T> struct addr_impl_ref {
  T & v_;
  inline addr_impl_ref( T & v ): v_( v ) {
  }
  inline operator T& () const {return v_;}
};
template<class T> struct addressof_impl {
  static inline T * f( T & v, long )     {
    return reinterpret_cast<T*>(&const_cast<char&>(reinterpret_cast<const volatile char &>(v)));
  }
};
template<class T> T * addressof( T & v ) {
  return addressof_impl<T>::f( addr_impl_ref<T>( v ), 0 );
}
void testRadar11487525_1(){
  bool s[25];
  addressof(s);
}

// radar://11487525 Don't crash on CK_LValueBitCast.
bool begin(double *it) {
  typedef bool type[25];
  bool *a = reinterpret_cast<type &>(*( reinterpret_cast<char *>( it )));
  return *a;
}

// radar://14164698 Don't crash on "assuming" a ComoundVal.
class JSONWireProtocolInputStream {
public:
  virtual ~JSONWireProtocolInputStream();
};
class JSONWireProtocolReader {
public:
  JSONWireProtocolReader(JSONWireProtocolInputStream& istream)
  : _istream{istream} {} // On evaluating a bind here,
                         // the dereference checker issues an assume on a CompoundVal.
~JSONWireProtocolReader();
private:
JSONWireProtocolInputStream& _istream;
};
class SocketWireProtocolStream : public JSONWireProtocolInputStream {
};
void test() {
  SocketWireProtocolStream stream{};
  JSONWireProtocolReader reader{stream};
}

// This crashed because the analyzer did not understand AttributedStmts.
void fallthrough() {
  switch (1) {
    case 1:
      [[clang::fallthrough]]; // expected-error {{does not directly precede}}
  }
}

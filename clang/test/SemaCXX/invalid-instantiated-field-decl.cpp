// RUN: %clang_cc1 -fsyntax-only -verify %s

template <typename T>
class SmallVectorImpl  {
public:
  explicit SmallVectorImpl(unsigned N) {
  }

  ~SmallVectorImpl() { }

};

template <typename T, unsigned N>
class SmallVector : public SmallVectorImpl<T> {
  typedef typename SmallVectorImpl<T>::U U; // expected-error {{no type named 'U' in 'SmallVectorImpl<CallSite>'}}
  enum {

    MinUs = (static_cast<unsigned int>(sizeof(T))*N +	// expected-error {{invalid application of 'sizeof' to an incomplete type 'CallSite'}}
             static_cast<unsigned int>(sizeof(U)) - 1) /
            static_cast<unsigned int>(sizeof(U)),
    NumInlineEltsElts = MinUs 
  };
  U InlineElts[NumInlineEltsElts];
public:
  SmallVector() : SmallVectorImpl<T>(NumInlineEltsElts) {
  }

};

class CallSite;	// expected-note {{forward declaration of 'CallSite'}}
class InlineFunctionInfo {
public:
  explicit InlineFunctionInfo() {}
  SmallVector<CallSite, 2> DevirtualizedCalls;	// expected-note {{in instantiation of template class 'SmallVector<CallSite, 2>' requested}}
};

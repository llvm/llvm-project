// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

@interface NSArray @end

@interface NSMutableArray : NSArray @end

@interface MyClass
{
  NSMutableArray * _array;
}

@property (readonly) NSMutableArray * array;

@end

@interface MyClass ()

@property (readwrite, retain) NSMutableArray * array;

@end

@implementation MyClass

@synthesize array=_array;

@end

int main(void)
{
  return 0;
}

class TCPPObject
{
public:
 TCPPObject(const TCPPObject& inObj);
 TCPPObject();
 ~TCPPObject();
 TCPPObject& operator=(const TCPPObject& inObj); // expected-note {{'operator=' declared here}}
private:
 void* fData;
};

class Trivial
{
public:
 Trivial(const Trivial& inObj);
 Trivial();
 ~Trivial();
private:
 void* fData;
};

@interface MyDocument
{
@private
 TCPPObject _cppObject;
 TCPPObject _ncppObject;
 Trivial _tcppObject;
}
@property (assign, readwrite) const TCPPObject& cppObject;
@property (assign, readwrite, nonatomic) const TCPPObject& ncppObject;
@property (assign, readwrite) const Trivial& tcppObject;
@end

@implementation MyDocument

@synthesize cppObject = _cppObject; // expected-error {{atomic property of reference type 'const TCPPObject &' cannot have non-trivial assignment operator}}
@synthesize ncppObject = _ncppObject;

@synthesize tcppObject = _tcppObject;
@end

struct IncompleteStruct; // expected-note 2 {{forward declaration of 'IncompleteStruct'}}
struct ConvertToIncomplete { operator IncompleteStruct&(); };
@interface SynthIncompleteRef
@property (readonly, nonatomic) IncompleteStruct& x; // expected-note {{property declared here}}
@property (readonly, nonatomic) IncompleteStruct& y; // expected-note {{property declared here}}
@end

@implementation SynthIncompleteRef // expected-error {{cannot synthesize property 'x' with incomplete type 'IncompleteStruct'}}
@synthesize y; // expected-error {{cannot synthesize property 'y' with incomplete type 'IncompleteStruct'}}
@end 


// Check error handling for instantiation during property synthesis.
template<typename T> class TemplateClass1 {
  T *x; // expected-error {{'x' declared as a pointer to a reference of type 'int &'}}
};
template<typename T> class TemplateClass2 {
  TemplateClass2& operator=(TemplateClass1<T>);
  TemplateClass2& operator=(TemplateClass2) { T(); } // expected-error {{reference to type 'int' requires an initializer}} \
                                                     // expected-note 2 {{implicitly declared private here}} \
                                                     // expected-note {{'operator=' declared here}}
};
__attribute__((objc_root_class)) @interface InterfaceWithTemplateProperties 
@property TemplateClass2<int&> intprop;
@property TemplateClass2<int&> &floatprop;
@end
@implementation InterfaceWithTemplateProperties // expected-error 2 {{'operator=' is a private member of 'TemplateClass2<int &>'}} \
																								// expected-error {{atomic property of reference type 'TemplateClass2<int &> &' cannot have non-trivial assignment operator}} \
																								// expected-note {{in instantiation of template class}} \
																								// expected-note {{in instantiation of member function}}
@end  

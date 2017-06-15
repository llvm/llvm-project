
#ifdef USE_NAMESPACE
namespace ns {
namespace ns2 {
#endif

#ifdef USE_ENCLOSING_RECORD
struct OuterRecord {
#endif

struct ClassInHeader {
  void pleaseImplement();
  void implemented();
  void pleaseImplementThisAsWell();
  void implementedToo();
  void anotherMethod();
};

#ifdef USE_ENCLOSING_RECORD
}
#endif

void ClassInHeader::anotherMethod() {
}
// CHECK: "{{.*}}classInHeader.h" "\n\nvoid ClassInHeader::pleaseImplement() { \n  <#code#>;\n}\n\nvoid ClassInHeader::pleaseImplementThisAsWell() { \n  <#code#>;\n}\n" [[@LINE-1]]:2 -> [[@LINE-1]]:2

#ifdef USE_NAMESPACE
}
}
#endif


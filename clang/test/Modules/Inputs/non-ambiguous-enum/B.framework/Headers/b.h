typedef long NSInteger;
typedef enum __attribute__((flag_enum,enum_extensibility(open))) MyObjCEnum : NSInteger MyObjCEnum;

enum MyObjCEnum : NSInteger {
    MyEnumCst = 1,
} __attribute__((availability(ios,introduced=11.0))) __attribute__((availability(tvos,unavailable))) ;

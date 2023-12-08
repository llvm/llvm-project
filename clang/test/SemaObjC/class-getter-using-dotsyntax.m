// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s
// expected-no-diagnostics

typedef struct objc_class *Class;

struct objc_class {
    Class isa;
};

typedef struct objc_object {
    Class isa;
} *id;

@interface XCActivityLogSection 
+ (unsigned)serializationFormatVersion;
+ (unsigned)sectionByDeserializingData;
+ (Class)retursClass;
@end

@implementation XCActivityLogSection

+ (unsigned)serializationFormatVersion
{

    return 0;
}
+ (unsigned)sectionByDeserializingData {
    unsigned version;
    return self.serializationFormatVersion;
}

+ (Class)retursClass {
    Class version;
    // FIXIT. (*version).isa does not work. Results in compiler error.
    return version->isa;
}

@end



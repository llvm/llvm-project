// RUN: %clang_cc1 -fsyntax-only -verify -Wno-objc-root-class %s

/*
  Conditions for warning:
  1. the property is atomic
  2. the current @implementation contains an @synthesize for the property
  3. the current @implementation contains a hand-written setter XOR getter
  4. the property is read-write
  
  Cases marked WARN should warn one the following:
  warning: Atomic property 'x' has a synthesized setter and a 
  manually-implemented getter, which may break atomicity.
  warning: Atomic property 'x' has a synthesized getter and a 
  manually-implemented setter, which may break atomicity.
  
  Cases not marked WARN only satisfy the indicated subset 
  of the conditions required to warn.

  There should be 8 warnings.
*/

@interface Foo 
{
    /* 12 4 */    int GetSet;
    /* WARN */    int Get;
    /* WARN */    int Set;
    /* 12 4 */    int None;
    /*  2 4 */    int GetSet_Nonatomic;
    /*  234 */    int Get_Nonatomic;
    /*  234 */    int Set_Nonatomic;
    /*  2 4 */    int None_Nonatomic;

    /* 12   */    int GetSet_ReadOnly;
    /* 123  */    int Get_ReadOnly;
    /* 123  */    int Set_ReadOnly;
    /* 12   */    int None_ReadOnly;
    /*  2   */    int GetSet_Nonatomic_ReadOnly;
    /*  23  */    int Get_Nonatomic_ReadOnly;
    /*  23  */    int Set_Nonatomic_ReadOnly;
    /*  2   */    int None_Nonatomic_ReadOnly;

    /* 12 4 */    int GetSet_ReadWriteInExt;
    /* WARN */    int Get_ReadWriteInExt;
    /* WARN */    int Set_ReadWriteInExt;
    /* 12 4 */    int None_ReadWriteInExt;
    /*  2 4 */    int GetSet_Nonatomic_ReadWriteInExt;
    /*  234 */    int Get_Nonatomic_ReadWriteInExt;
    /*  234 */    int Set_Nonatomic_ReadWriteInExt;
    /*  2 4 */    int None_Nonatomic_ReadWriteInExt;


    /* 12 4 */    int GetSet_LateSynthesize;
    /* WARN */    int Get_LateSynthesize;
    /* WARN */    int Set_LateSynthesize;
    /* 12 4 */    int None_LateSynthesize;
    /*  2 4 */    int GetSet_Nonatomic_LateSynthesize;
    /*  234 */    int Get_Nonatomic_LateSynthesize;
    /*  234 */    int Set_Nonatomic_LateSynthesize;
    /*  2 4 */    int None_Nonatomic_LateSynthesize;

    /* 12   */    int GetSet_ReadOnly_LateSynthesize;
    /* 123  */    int Get_ReadOnly_LateSynthesize;
    /* 123  */    int Set_ReadOnly_LateSynthesize;
    /* 12   */    int None_ReadOnly_LateSynthesize;
    /*  2   */    int GetSet_Nonatomic_ReadOnly_LateSynthesize;
    /*  23  */    int Get_Nonatomic_ReadOnly_LateSynthesize;
    /*  23  */    int Set_Nonatomic_ReadOnly_LateSynthesize;
    /*  2   */    int None_Nonatomic_ReadOnly_LateSynthesize;

    /* 12 4 */    int GetSet_ReadWriteInExt_LateSynthesize;
    /* WARN */    int Get_ReadWriteInExt_LateSynthesize;
    /* WARN */    int Set_ReadWriteInExt_LateSynthesize;
    /* 12 4 */    int None_ReadWriteInExt_LateSynthesize;
    /*  2 4 */    int GetSet_Nonatomic_ReadWriteInExt_LateSynthesize;
    /*  234 */    int Get_Nonatomic_ReadWriteInExt_LateSynthesize;
    /*  234 */    int Set_Nonatomic_ReadWriteInExt_LateSynthesize;
    /*  2 4 */    int None_Nonatomic_ReadWriteInExt_LateSynthesize;


    /* 1  4 */    int GetSet_NoSynthesize;
    /* 1 34 */    int Get_NoSynthesize;
    /* 1 34 */    int Set_NoSynthesize;
    /* 1  4 */    int None_NoSynthesize;
    /*    4 */    int GetSet_Nonatomic_NoSynthesize;
    /*   34 */    int Get_Nonatomic_NoSynthesize;
    /*   34 */    int Set_Nonatomic_NoSynthesize;
    /*    4 */    int None_Nonatomic_NoSynthesize;

    /* 1    */    int GetSet_ReadOnly_NoSynthesize;
    /* 1 3  */    int Get_ReadOnly_NoSynthesize;
    /* 1 3  */    int Set_ReadOnly_NoSynthesize;
    /* 1    */    int None_ReadOnly_NoSynthesize;
    /*      */    int GetSet_Nonatomic_ReadOnly_NoSynthesize;
    /*   3  */    int Get_Nonatomic_ReadOnly_NoSynthesize;
    /*   3  */    int Set_Nonatomic_ReadOnly_NoSynthesize;
    /*      */    int None_Nonatomic_ReadOnly_NoSynthesize;

    /* 1  4 */    int GetSet_ReadWriteInExt_NoSynthesize;
    /* 1 34 */    int Get_ReadWriteInExt_NoSynthesize;
    /* 1 34 */    int Set_ReadWriteInExt_NoSynthesize;
    /* 1  4 */    int None_ReadWriteInExt_NoSynthesize;
    /*    4 */    int GetSet_Nonatomic_ReadWriteInExt_NoSynthesize;
    /*   34 */    int Get_Nonatomic_ReadWriteInExt_NoSynthesize;
    /*   34 */    int Set_Nonatomic_ReadWriteInExt_NoSynthesize;
    /*    4 */    int None_Nonatomic_ReadWriteInExt_NoSynthesize;
}

// read-write - might warn
@property int GetSet;
@property int Get;	// expected-note {{property declared here}} \
                        // expected-note {{setter and getter must both be synthesized}}
@property int Set;	// expected-note {{property declared here}} \
                        // expected-note {{setter and getter must both be synthesized}}
@property int None;
@property(nonatomic) int GetSet_Nonatomic;
@property(nonatomic) int Get_Nonatomic;
@property(nonatomic) int Set_Nonatomic;
@property(nonatomic) int None_Nonatomic;

// read-only - must not warn
@property(readonly) int GetSet_ReadOnly;
@property(readonly) int Get_ReadOnly;
@property(readonly) int Set_ReadOnly;
@property(readonly) int None_ReadOnly;
@property(nonatomic,readonly) int GetSet_Nonatomic_ReadOnly;
@property(nonatomic,readonly) int Get_Nonatomic_ReadOnly;
@property(nonatomic,readonly) int Set_Nonatomic_ReadOnly;
@property(nonatomic,readonly) int None_Nonatomic_ReadOnly;

// read-only in class, read-write in class extension - might warn
@property(readonly) int GetSet_ReadWriteInExt;
@property(readonly) int Get_ReadWriteInExt;	// expected-note {{property declared here}} \
                                                // expected-note {{setter and getter must both be synthesized}}
@property(readonly) int Set_ReadWriteInExt;	// expected-note {{property declared here}} \
                                                // expected-note {{setter and getter must both be synthesized}}
@property(readonly) int None_ReadWriteInExt;
@property(nonatomic,readonly) int GetSet_Nonatomic_ReadWriteInExt;
@property(nonatomic,readonly) int Get_Nonatomic_ReadWriteInExt;
@property(nonatomic,readonly) int Set_Nonatomic_ReadWriteInExt;
@property(nonatomic,readonly) int None_Nonatomic_ReadWriteInExt;


// same as above, but @synthesize follows the hand-written methods - might warn
@property int GetSet_LateSynthesize;
@property int Get_LateSynthesize;	// expected-note {{property declared here}} \
                                        // expected-note {{setter and getter must both be synthesized}}
@property int Set_LateSynthesize;	// expected-note {{property declared here}} \
                                        // expected-note {{setter and getter must both be synthesized}}
@property int None_LateSynthesize;
@property(nonatomic) int GetSet_Nonatomic_LateSynthesize;
@property(nonatomic) int Get_Nonatomic_LateSynthesize;
@property(nonatomic) int Set_Nonatomic_LateSynthesize;
@property(nonatomic) int None_Nonatomic_LateSynthesize;

@property(readonly) int GetSet_ReadOnly_LateSynthesize;
@property(readonly) int Get_ReadOnly_LateSynthesize;
@property(readonly) int Set_ReadOnly_LateSynthesize;
@property(readonly) int None_ReadOnly_LateSynthesize;
@property(nonatomic,readonly) int GetSet_Nonatomic_ReadOnly_LateSynthesize;
@property(nonatomic,readonly) int Get_Nonatomic_ReadOnly_LateSynthesize;
@property(nonatomic,readonly) int Set_Nonatomic_ReadOnly_LateSynthesize;
@property(nonatomic,readonly) int None_Nonatomic_ReadOnly_LateSynthesize;

@property(readonly) int GetSet_ReadWriteInExt_LateSynthesize;
@property(readonly) int Get_ReadWriteInExt_LateSynthesize;	// expected-note {{property declared here}} \
                                                                // expected-note {{setter and getter must both be synthesized}}
@property(readonly) int Set_ReadWriteInExt_LateSynthesize;	// expected-note {{property declared here}} \
                                                                // expected-note {{setter and getter must both be synthesized}}
@property(readonly) int None_ReadWriteInExt_LateSynthesize;
@property(nonatomic,readonly) int GetSet_Nonatomic_ReadWriteInExt_LateSynthesize;
@property(nonatomic,readonly) int Get_Nonatomic_ReadWriteInExt_LateSynthesize;
@property(nonatomic,readonly) int Set_Nonatomic_ReadWriteInExt_LateSynthesize;
@property(nonatomic,readonly) int None_Nonatomic_ReadWriteInExt_LateSynthesize;


// same as above, but with no @synthesize - must not warn
@property int GetSet_NoSynthesize;
@property int Get_NoSynthesize;
@property int Set_NoSynthesize;
@property int None_NoSynthesize;
@property(nonatomic) int GetSet_Nonatomic_NoSynthesize;
@property(nonatomic) int Get_Nonatomic_NoSynthesize;
@property(nonatomic) int Set_Nonatomic_NoSynthesize;
@property(nonatomic) int None_Nonatomic_NoSynthesize;

@property(readonly) int GetSet_ReadOnly_NoSynthesize;
@property(readonly) int Get_ReadOnly_NoSynthesize;
@property(readonly) int Set_ReadOnly_NoSynthesize;
@property(readonly) int None_ReadOnly_NoSynthesize;
@property(nonatomic,readonly) int GetSet_Nonatomic_ReadOnly_NoSynthesize;
@property(nonatomic,readonly) int Get_Nonatomic_ReadOnly_NoSynthesize;
@property(nonatomic,readonly) int Set_Nonatomic_ReadOnly_NoSynthesize;
@property(nonatomic,readonly) int None_Nonatomic_ReadOnly_NoSynthesize;

@property(readonly) int GetSet_ReadWriteInExt_NoSynthesize;
@property(readonly) int Get_ReadWriteInExt_NoSynthesize;
@property(readonly) int Set_ReadWriteInExt_NoSynthesize;
@property(readonly) int None_ReadWriteInExt_NoSynthesize;
@property(nonatomic,readonly) int GetSet_Nonatomic_ReadWriteInExt_NoSynthesize;
@property(nonatomic,readonly) int Get_Nonatomic_ReadWriteInExt_NoSynthesize;
@property(nonatomic,readonly) int Set_Nonatomic_ReadWriteInExt_NoSynthesize;
@property(nonatomic,readonly) int None_Nonatomic_ReadWriteInExt_NoSynthesize;

@end


@interface Foo ()

@property(readwrite) int GetSet_ReadWriteInExt;
@property(readwrite) int Get_ReadWriteInExt;
@property(readwrite) int Set_ReadWriteInExt;
@property(readwrite) int None_ReadWriteInExt;
@property(nonatomic,readwrite) int GetSet_Nonatomic_ReadWriteInExt;
@property(nonatomic,readwrite) int Get_Nonatomic_ReadWriteInExt;
@property(nonatomic,readwrite) int Set_Nonatomic_ReadWriteInExt;
@property(nonatomic,readwrite) int None_Nonatomic_ReadWriteInExt;

@property(readwrite) int GetSet_ReadWriteInExt_LateSynthesize;
@property(readwrite) int Get_ReadWriteInExt_LateSynthesize;
@property(readwrite) int Set_ReadWriteInExt_LateSynthesize;
@property(readwrite) int None_ReadWriteInExt_LateSynthesize;
@property(nonatomic,readwrite) int GetSet_Nonatomic_ReadWriteInExt_LateSynthesize;
@property(nonatomic,readwrite) int Get_Nonatomic_ReadWriteInExt_LateSynthesize;
@property(nonatomic,readwrite) int Set_Nonatomic_ReadWriteInExt_LateSynthesize;
@property(nonatomic,readwrite) int None_Nonatomic_ReadWriteInExt_LateSynthesize;

@property(readwrite) int GetSet_ReadWriteInExt_NoSynthesize;
@property(readwrite) int Get_ReadWriteInExt_NoSynthesize;
@property(readwrite) int Set_ReadWriteInExt_NoSynthesize;
@property(readwrite) int None_ReadWriteInExt_NoSynthesize;
@property(nonatomic,readwrite) int GetSet_Nonatomic_ReadWriteInExt_NoSynthesize;
@property(nonatomic,readwrite) int Get_Nonatomic_ReadWriteInExt_NoSynthesize;
@property(nonatomic,readwrite) int Set_Nonatomic_ReadWriteInExt_NoSynthesize;
@property(nonatomic,readwrite) int None_Nonatomic_ReadWriteInExt_NoSynthesize;

@end

@implementation Foo

@synthesize GetSet, Get, Set, None, GetSet_Nonatomic, Get_Nonatomic, Set_Nonatomic, None_Nonatomic;
@synthesize GetSet_ReadOnly, Get_ReadOnly, Set_ReadOnly, None_ReadOnly, GetSet_Nonatomic_ReadOnly, Get_Nonatomic_ReadOnly, Set_Nonatomic_ReadOnly, None_Nonatomic_ReadOnly;
@synthesize GetSet_ReadWriteInExt, Get_ReadWriteInExt, Set_ReadWriteInExt, None_ReadWriteInExt, GetSet_Nonatomic_ReadWriteInExt, Get_Nonatomic_ReadWriteInExt, Set_Nonatomic_ReadWriteInExt, None_Nonatomic_ReadWriteInExt;

#define GET(x) \
    -(int) x { return self->x; }  
#define SET(x) \
    -(void) set##x:(int)value { self->x = value; }  

GET(GetSet)
SET(GetSet)
GET(Get) // expected-warning {{writable atomic property 'Get' cannot pair a synthesized setter with a user defined getter}} 
SET(Set) // expected-warning {{writable atomic property 'Set' cannot pair a synthesized getter with a user defined setter}}
GET(GetSet_Nonatomic)
SET(GetSet_Nonatomic)
GET(Get_Nonatomic)
SET(Set_Nonatomic)

GET(GetSet_ReadOnly)
SET(GetSet_ReadOnly)
GET(Get_ReadOnly)
SET(Set_ReadOnly)
GET(GetSet_Nonatomic_ReadOnly)
SET(GetSet_Nonatomic_ReadOnly)
GET(Get_Nonatomic_ReadOnly)
SET(Set_Nonatomic_ReadOnly)

GET(GetSet_ReadWriteInExt)
SET(GetSet_ReadWriteInExt)
GET(Get_ReadWriteInExt) // expected-warning {{writable atomic property 'Get_ReadWriteInExt' cannot pair a synthesized setter with a user defined getter}} 
SET(Set_ReadWriteInExt) // expected-warning {{writable atomic property 'Set_ReadWriteInExt' cannot pair a synthesized getter with a user defined setter}}
GET(GetSet_Nonatomic_ReadWriteInExt)
SET(GetSet_Nonatomic_ReadWriteInExt)
GET(Get_Nonatomic_ReadWriteInExt)
SET(Set_Nonatomic_ReadWriteInExt)


GET(GetSet_LateSynthesize)
SET(GetSet_LateSynthesize)
GET(Get_LateSynthesize) // expected-warning {{writable atomic property 'Get_LateSynthesize' cannot pair a synthesized setter with a user defined getter}} 
SET(Set_LateSynthesize) // expected-warning {{writable atomic property 'Set_LateSynthesize' cannot pair a synthesized getter with a user defined setter}}
GET(GetSet_Nonatomic_LateSynthesize)
SET(GetSet_Nonatomic_LateSynthesize)
GET(Get_Nonatomic_LateSynthesize)
SET(Set_Nonatomic_LateSynthesize)

GET(GetSet_ReadOnly_LateSynthesize)
SET(GetSet_ReadOnly_LateSynthesize)
GET(Get_ReadOnly_LateSynthesize)
SET(Set_ReadOnly_LateSynthesize)
GET(GetSet_Nonatomic_ReadOnly_LateSynthesize)
SET(GetSet_Nonatomic_ReadOnly_LateSynthesize)
GET(Get_Nonatomic_ReadOnly_LateSynthesize)
SET(Set_Nonatomic_ReadOnly_LateSynthesize)

GET(GetSet_ReadWriteInExt_LateSynthesize)
SET(GetSet_ReadWriteInExt_LateSynthesize)
GET(Get_ReadWriteInExt_LateSynthesize) // expected-warning {{writable atomic property 'Get_ReadWriteInExt_LateSynthesize' cannot pair a synthesized setter with a user defined getter}}
SET(Set_ReadWriteInExt_LateSynthesize) // expected-warning {{writable atomic property 'Set_ReadWriteInExt_LateSynthesize' cannot pair a synthesized getter with a user defined setter}}
GET(GetSet_Nonatomic_ReadWriteInExt_LateSynthesize)
SET(GetSet_Nonatomic_ReadWriteInExt_LateSynthesize)
GET(Get_Nonatomic_ReadWriteInExt_LateSynthesize)
SET(Set_Nonatomic_ReadWriteInExt_LateSynthesize)


GET(GetSet_NoSynthesize)
SET(GetSet_NoSynthesize)
GET(Get_NoSynthesize)
SET(Set_NoSynthesize)
GET(GetSet_Nonatomic_NoSynthesize)
SET(GetSet_Nonatomic_NoSynthesize)
GET(Get_Nonatomic_NoSynthesize)
SET(Set_Nonatomic_NoSynthesize)

GET(GetSet_ReadOnly_NoSynthesize)
SET(GetSet_ReadOnly_NoSynthesize)
GET(Get_ReadOnly_NoSynthesize)
SET(Set_ReadOnly_NoSynthesize)
GET(GetSet_Nonatomic_ReadOnly_NoSynthesize)
SET(GetSet_Nonatomic_ReadOnly_NoSynthesize)
GET(Get_Nonatomic_ReadOnly_NoSynthesize)
SET(Set_Nonatomic_ReadOnly_NoSynthesize)

GET(GetSet_ReadWriteInExt_NoSynthesize)
SET(GetSet_ReadWriteInExt_NoSynthesize)
GET(Get_ReadWriteInExt_NoSynthesize)
SET(Set_ReadWriteInExt_NoSynthesize)
GET(GetSet_Nonatomic_ReadWriteInExt_NoSynthesize)
SET(GetSet_Nonatomic_ReadWriteInExt_NoSynthesize)
GET(Get_Nonatomic_ReadWriteInExt_NoSynthesize)
SET(Set_Nonatomic_ReadWriteInExt_NoSynthesize)


// late synthesize - follows getter/setter implementations

@synthesize GetSet_LateSynthesize, Get_LateSynthesize, Set_LateSynthesize, None_LateSynthesize, GetSet_Nonatomic_LateSynthesize, Get_Nonatomic_LateSynthesize, Set_Nonatomic_LateSynthesize, None_Nonatomic_LateSynthesize;
@synthesize GetSet_ReadOnly_LateSynthesize, Get_ReadOnly_LateSynthesize, Set_ReadOnly_LateSynthesize, None_ReadOnly_LateSynthesize, GetSet_Nonatomic_ReadOnly_LateSynthesize, Get_Nonatomic_ReadOnly_LateSynthesize, Set_Nonatomic_ReadOnly_LateSynthesize, None_Nonatomic_ReadOnly_LateSynthesize;
@synthesize GetSet_ReadWriteInExt_LateSynthesize, Get_ReadWriteInExt_LateSynthesize, Set_ReadWriteInExt_LateSynthesize, None_ReadWriteInExt_LateSynthesize, GetSet_Nonatomic_ReadWriteInExt_LateSynthesize, Get_Nonatomic_ReadWriteInExt_LateSynthesize, Set_Nonatomic_ReadWriteInExt_LateSynthesize, None_Nonatomic_ReadWriteInExt_LateSynthesize;

// no synthesize - use dynamic instead

@dynamic GetSet_NoSynthesize, Get_NoSynthesize, Set_NoSynthesize, None_NoSynthesize, GetSet_Nonatomic_NoSynthesize, Get_Nonatomic_NoSynthesize, Set_Nonatomic_NoSynthesize, None_Nonatomic_NoSynthesize;
@dynamic GetSet_ReadOnly_NoSynthesize, Get_ReadOnly_NoSynthesize, Set_ReadOnly_NoSynthesize, None_ReadOnly_NoSynthesize, GetSet_Nonatomic_ReadOnly_NoSynthesize, Get_Nonatomic_ReadOnly_NoSynthesize, Set_Nonatomic_ReadOnly_NoSynthesize, None_Nonatomic_ReadOnly_NoSynthesize;
@dynamic GetSet_ReadWriteInExt_NoSynthesize, Get_ReadWriteInExt_NoSynthesize, Set_ReadWriteInExt_NoSynthesize, None_ReadWriteInExt_NoSynthesize, GetSet_Nonatomic_ReadWriteInExt_NoSynthesize, Get_Nonatomic_ReadWriteInExt_NoSynthesize, Set_Nonatomic_ReadWriteInExt_NoSynthesize, None_Nonatomic_ReadWriteInExt_NoSynthesize;

@end

/*
// the following method should cause a warning along the lines of
// :warning: Atomic property 'x' cannot pair a synthesized setter/getter with a manually implemented setter/getter
- (void) setX: (int) aValue
{
    x = aValue;
}

// no warning 'cause this is nonatomic
- (void) setY: (int) aValue
{
    y = aValue;
}

// the following method should cause a warning along the lines of
// :warning: Atomic property 'x' cannot pair a synthesized setter/getter with a manually implemented setter/getter
- (int) j
{
    return j;
}

// no warning 'cause this is nonatomic
- (int) k
{
    return k;
}
@end
*/
int main (int argc, const char * argv[]) {
    return 0;
}

@import LayeredKitImpl;

// @interface declarations already don't inherit attributes from forward 
// declarations, so in order to test this properly we have to /not/ define
// UpwardClass anywhere.

// @interface UpwardClass
// @end

@protocol UpwardProto
@end

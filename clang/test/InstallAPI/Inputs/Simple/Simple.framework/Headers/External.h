#import <Foundation/Foundation.h>

// Sub-class an external defined ObjC Class.
@interface ExternalManagedObject : NSManagedObject
- (void)foo;
@end

// Add category to external defined ObjC Class.
@interface NSManagedObject (Simple)
- (int)supportsSimple;
@end

// CoreData Accessors are dynamically generated and have no implementation.
@interface ExternalManagedObject (CoreDataGeneratedAccessors)
- (void)addChildObject:(ExternalManagedObject *)value;
- (void)removeChildObject:(ExternalManagedObject *)value;
- (void)addChild:(NSSet *)values;
- (void)removeChild:(NSSet *)values;
@end

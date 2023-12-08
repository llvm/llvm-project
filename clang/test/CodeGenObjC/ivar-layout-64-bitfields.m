// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc -emit-llvm -o %t %s
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -fobjc-gc -emit-llvm -o %t %s

#ifdef __cplusplus
typedef bool _Bool;
#endif

@interface I
{
  struct {
    unsigned int d : 1;
  } bitfield;
}
@end

@implementation I
@end

@interface J
{
    struct {
        unsigned short _reserved : 16;

        _Bool _draggedNodesAreDeletable: 1;
        _Bool _draggedOutsideOutlineView : 1;
        _Bool _adapterRespondsTo_addRootPaths : 1;
        _Bool _adapterRespondsTo_moveDataNodes : 1;
        _Bool _adapterRespondsTo_removeRootDataNode : 1;
        _Bool _adapterRespondsTo_doubleClickDataNode : 1;
        _Bool _adapterRespondsTo_selectDataNode : 1;
        _Bool _adapterRespondsTo_textDidEndEditing : 1;

        _Bool _adapterRespondsTo_updateAndSaveRoots : 1;
        _Bool _adapterRespondsTo_askToDeleteRootNodes : 1;
        _Bool _adapterRespondsTo_contextMenuForSelectedNodes : 1;
        _Bool _adapterRespondsTo_pasteboardFilenamesForNodes : 1;
        _Bool _adapterRespondsTo_writeItemsToPasteboard : 1;
        _Bool _adapterRespondsTo_writeItemsToPasteboardXXXX : 1;
    } _flags;
}
@end

@implementation J
@end



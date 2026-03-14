#ifndef B1_H
#define B1_H
typedef struct {} foo_t;
typedef foo_t foo2_t;
#include "a2.h"
namespace llvm {
class MachineBasicBlock;
template <class NodeT> class DomTreeNodeBase;
typedef DomTreeNodeBase<MachineBasicBlock> MachineDomTreeNode;
}
#endif

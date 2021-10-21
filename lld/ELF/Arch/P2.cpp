//===- P2.cpp ------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// generally, programs will be creates as follows:
//
//   ld.lld -Ttext=0 -o foo foo.o
//   objcopy -O binary --only-section=.text foo output.bin
//
// Note that the current P2 support is very preliminary so you can't
// link any useful program yet, though.
//
//===----------------------------------------------------------------------===//

#include "InputFiles.h"
#include "Symbols.h"
#include "Target.h"
#include "lld/Common/ErrorHandler.h"
#include "llvm/Object/ELF.h"
#include "llvm/Support/Endian.h"

#define DEBUG_TYPE "p2"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::support::endian;
using namespace llvm::ELF;

namespace lld {
    namespace elf {

        namespace {
            class P2 final : public TargetInfo {
            public:
                RelExpr getRelExpr(RelType type, const Symbol &s, const uint8_t *loc) const override;
                void relocate(uint8_t *loc, const Relocation &rel, uint64_t val) const override;
            };
        } // namespace

        RelExpr P2::getRelExpr(RelType type, const Symbol &s, const uint8_t *loc) const {
            switch (type) {
                default:
                    return R_ABS;
                case R_P2_32:
                case R_P2_20:
                case R_P2_AUG20:
                case R_P2_COG9:
                    return R_ABS;
                case R_P2_PC20:
                    return R_PC;
            }
        }

        void P2::relocate(uint8_t *loc, const Relocation &rel, uint64_t val) const {

            if (rel.sym->isLocal()) {
                LLVM_DEBUG(outs() << "symbol is local\n");
            }

            LLVM_DEBUG(outs() << "relocate: " << rel.sym->getName() << "\n");
            LLVM_DEBUG(outs() << "reloc value is " << (int)val << "\n");

            switch (rel.type) {
                case R_P2_32: {
                    write32le(loc, val);
                    break;
                }
                case R_P2_20: {
                    uint32_t inst = read32le(loc);
                    inst += val & 0xfffff; // val is the relocation value, which is just the section this relocation references
                                           // so, we just add the new value (val) to the instructions to offset the 20 bit operand
                    write32le(loc, inst);
                    break;
                }
                case R_P2_AUG20: {
                    // special relocation where we modify 2 instructions to perform an immediate load of a 20-bit (or greater) immediate.
                    // by invoking the augd or augs instruction
                    uint32_t inst = read32le(loc);
                    uint32_t aug = read32le(loc-4) & ~0x7fffff; // the previous instruction is expected to be an AUGS/D

                    //LLVM_DEBUG(outs() << "reloc offset: " << rel.offset << "\n");
                    LLVM_DEBUG(outs() << "reloc value is " << (int)val << "\n");
                    //LLVM_DEBUG(outs() << "adjusted value is " << (int)(val & 0x1ff) << "\n");

                    inst += val & 0x1ff; // get the lower 9 bits into the current instruction
                    aug |= (val >> 9); // get the upper 23 bits into the previous AUG instruction

                    write32le(loc-4, aug);
                    write32le(loc, inst);
                    break;
                }
                case R_P2_COG9: {
                    uint32_t inst = read32le(loc);
                    // TODO: make this more flexible. right now it assumes the COG-based library lives at
                    // 0x100. Eventually we want to mark any function/variable as being able to live in the cog.
                    inst += ((val-0x100)/4) & 0x1ff;
                    LLVM_DEBUG(outs() << "cog function relocation\n");
                    LLVM_DEBUG(outs() << "original value is " << (int)val << "\n");
                    LLVM_DEBUG(outs() << "adjusted value is " << (int)(((val-0x100)/4) & 0x1ff) << "\n");
                    write32le(loc, inst);
                    break;
                }
                default:
                    error(getErrorLocation(loc) + "unrecognized relocation " + toString(rel.type));
            }
        }

        TargetInfo *getP2TargetInfo() {
            static P2 target;
            return &target;
        }
    } // namespace elf
} // namespace lld

#include "llvm/Pass.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "base_def.hpp"

#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <queue>

/**
 * References : https://github.com/jarulraj/llvm/, https://llvm.org/doxygen/LiveVariables_8cpp_source.html
 * This header contains all the class & function implementations for the live variable analysis. 
*/

namespace llvm
{

    template <class T>
    void print_container(std::set<T> s)
    {
        errs() << " {";
        for (const auto &elems : s)
        {
            errs() << elems << " ";
        }
        errs() << "}.";
    }

    // convert an expression to index in Domain D.
    int domainIndex(std::vector<void *> &D, void *ptr);

    // convert LLVM value to corresponding std::string
    std::string getValueName(Value *v);

    // find and return index i in Domain set.
    int Dataflow::domainIndex(void *ptr)
    {

        std::vector<void *> &D = this->domain;

        auto it = std::find(D.begin(), D.end(), ptr);
        int i = std::distance(D.begin(), it);

        if (i >= this->domain.size() || i < 0)
        {
            i = INDEX_NOT_FOUND;
        }

        return i;
    }

    void error(std::string err)
    {
        errs() << "\n[!ERROR!] " << err << "\n";
        exit(-1);
    }

    // store it to a cutomize type
    Expression::Expression(Instruction *inst)
    {
        if (auto bin_op = dyn_cast<BinaryOperator>(inst))
        {
            this->op = bin_op->getOpcode();
            this->v1 = bin_op->getOperand(0);
            this->v2 = bin_op->getOperand(1);
        }
    }

    bool Expression::operator==(const Expression &expr)
    {
        return (expr.op == this->op && expr.v1 == this->v1 && expr.v2 == this->v2);
    }

    // meet operation on sets.
    std::set<int> Dataflow::MeetOperationSets(std::vector<std::set<int>> in_sets)
    {
        std::set<int> result;

        if (in_sets.empty())
        {
            return result;
        }

        for (auto in : in_sets)
        {
            // this is a plain union over all sets. (duplicates not added again).
            result.insert(in.begin(), in.end());
        }

        return result;
    }

    std::map<BasicBlock *, BlockResult> Dataflow::pre_analysis(Function &F, std::set<int> boundary, std::set<int> interior)
    {

        std::map<BasicBlock *, BlockResult> result;
        std::vector<void *> &domain = this->domain;
        std::set<int> base;

        // initialize the first Block we need to iterate accoring to direction
        std::vector<BasicBlock *> initList, traverseList;

        // For each basic block BB in Function F.
        for (auto &BB : F)
        {
            if (isa<ReturnInst>(BB.getTerminator()))
            {
                initList.push_back(&BB);
            }
        }

        // push other blocks into the list
        std::map<BasicBlock *, std::vector<BasicBlock *>> NextSuccesorBlocks;
        for (auto &BB : F)
        {
            for (auto succ_BB = succ_begin(&BB); succ_BB != succ_end(&BB); ++succ_BB)
            {
                NextSuccesorBlocks[&BB].push_back(*succ_BB);
            }
        }

        // initialize boudary set value.
        BlockResult boundaryRes = BlockResult();
        boundaryRes.out = boundary;

        for (auto BB : initList)
        {
            result.insert(std::pair<BasicBlock *, BlockResult>(BB, boundaryRes));
        }

        // initalize interior set value
        BlockResult interiorRes = BlockResult();

        interiorRes.in = interior;
        base = interior;

        for (auto &BB : F)
        {
            if (result.find(&BB) == result.end())
            {
                result.insert(std::pair<BasicBlock *, BlockResult>(&BB, interiorRes));
            }
        }

        std::set<BasicBlock *> visited;
        while (!initList.empty())
        {
            BasicBlock *currBB = initList[0];
            traverseList.push_back(currBB);
            initList.erase(initList.begin());
            visited.insert(currBB);

            for (auto prev_BB = pred_begin(currBB); prev_BB != pred_end(currBB); ++prev_BB)
            {
                if (visited.find(*prev_BB) == visited.end())
                {
                    initList.push_back(*prev_BB);
                }
            }
        }

        // fixed point algorithm, iterate until it does not change in simple bitvector analysis.
        bool converged = false;
        while (!converged)
        {
            converged = true;

            for (auto currBB : traverseList)
            {
                // we use calculate meet value first
                std::vector<std::set<int>> meetInput;

                // if we have to initialize with some values
                if (isa<ReturnInst>(currBB->getTerminator()))
                {
                    meetInput.push_back(base);
                }

                for (auto n : NextSuccesorBlocks[currBB])
                {
                    std::set<int> value;
                    value = result[n].in;
                    meetInput.push_back(value);
                }

                // then is transfer value
                std::set<int> meetResult = MeetOperationSets(meetInput);
                result[currBB].out = meetResult;

                std::set<int> *blockInput = &result[currBB].out;
                BlockTransferFunction transferRes = transferFn(*blockInput, currBB);
                std::set<int> *blockOutput = &result[currBB].in;

                // check if previous result and the transfer result are the same, waiting for fixed point.
                if (converged)
                {
                    if (transferRes.transfer != *blockOutput ||
                        result[currBB].transferOutput.neighbor.size() != transferRes.neighbor.size())
                    {
                        converged = false;
                    }
                }

                // update value
                *blockOutput = transferRes.transfer;
                result[currBB].transferOutput.neighbor = transferRes.neighbor;
            }
        }

        std::map<BasicBlock *, BlockResult> analysis;
        analysis = result;

        return analysis;
    }

    // The following code is adapted from https://github.com/jarulraj/llvm/ .

    std::string getValueName(Value *val)
    {

        // If we can get name directly
        if (val->getName().str().length() > 0)
        {

            return "%" + val->getName().str();
        }
        else if (isa<Instruction>(val))
        {

            std::string str = "";
            raw_string_ostream *strm = new raw_string_ostream(str);
            val->print(*strm);

            std::string inst = strm->str();
            size_t idx1 = inst.find("%");
            size_t idx2 = inst.find(" ", idx1);

            if (idx1 != std::string::npos && idx2 != std::string::npos && idx1 == 2)
            {
                return inst.substr(idx1, idx2 - idx1);
            }
            else
            {
                return "";
            }
        }
        else if (ConstantInt *consts = dyn_cast<ConstantInt>(val))
        {

            std::string str = "";
            raw_string_ostream *strm = new raw_string_ostream(str);
            consts->getValue().print(*strm, true);
            return strm->str();
        }
        else
        {

            std::string str = "";
            raw_string_ostream *strm = new raw_string_ostream(str);
            val->print(*strm);
            std::string inst = strm->str();
            return "\"" + inst + "\"";
        }
    }
    class LiveVariableAnalysis : public Dataflow
    {
    public:
        LiveVariableAnalysis(std::vector<void *> domain)
            : Dataflow(domain) {}

        // IN(S) = (OUT(S) - KILL(S)) U GEN(S);
        BlockTransferFunction transferFn(std::set<int> OUT, BasicBlock *curr)
        {

            BlockTransferFunction output;
            std::set<int> gen, kill; // GEN and KILL

            // def-use chain updation for PHINodes and related Blocks.
            for (auto inst = curr->begin(); inst != curr->end(); ++inst)
            {
                if (PHINode *phi = dyn_cast<PHINode>(&*inst))
                {
                    for (int i = 0; i < phi->getNumIncomingValues(); ++i)
                    {
                        auto val = phi->getIncomingValue(i);
                        if (isa<Instruction>(val) || isa<Argument>(val))
                        {

                            // add instructions to neighbor as we did in class, we transfer the instr to pred blocks after modification.
                            auto incomingBlock = phi->getIncomingBlock(i);
                            if (output.neighbor.find(incomingBlock) == output.neighbor.end())
                            {
                                output.neighbor.insert(std::pair<BasicBlock *, std::set<int>>(incomingBlock, std::set<int>()));
                            }
                            int index = domainIndex(val);
                            output.neighbor[incomingBlock].insert(index);
                        }
                    }
                }
                else
                {
                    for (auto op = inst->op_begin(); op != inst->op_end(); ++op)
                    {
                        Value *val = *op;
                        if (isa<Instruction>(val) || isa<Argument>(val))
                        {

                            // if previous defined and used now, add to use set
                            int index = domainIndex(val);
                            if (index != INDEX_NOT_FOUND)
                            {
                                if (kill.find(index) == kill.end())
                                {
                                    gen.insert(index);
                                }
                            }
                        }
                    }
                }

                int i = domainIndex(&*inst);
                if (i != INDEX_NOT_FOUND)
                {
                    kill.insert(i);
                }
            }

            auto tmp = diff_set(OUT, kill);        // OUT(S) - KILL(S)
            output.transfer = union_set(gen, tmp); // IN(S)
            return output;
        }
    };
}; // namespace llvm
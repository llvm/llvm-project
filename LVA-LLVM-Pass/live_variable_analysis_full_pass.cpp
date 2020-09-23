#include "headers/lva_header.hpp"

/**
 * References : https://github.com/jarulraj/llvm/, https://llvm.org/doxygen/LiveVariables_8cpp_source.html
 * LVA Analysis on full file. 
 */

using namespace llvm;

long long int counter = 0, inst = 0, iteration = 0;                  // global counters.
std::vector<std::set<std::string>> liveSetElems;                     // Store the live vars sets at each program point.
std::vector<std::pair<Instruction *, std::set<std::string>>> output; // Stores the processed output.
std::unordered_map<int, int> histogram;                              // Histogram mapping.

namespace
{
    class Liveness : public FunctionPass
    {
    public:
        static char ID;

        virtual void getAnalysisUsage(AnalysisUsage &AU) const
        {
            AU.setPreservesAll();
        }

        Liveness() : FunctionPass(ID) {} // Intraprocedural only. not ModulePass.

        virtual bool runOnFunction(Function &F)
        {
            // errs() << "Function: " << F.getName() << "\n";
            std::map<BasicBlock *, BlockResult> fresult; // Store BB final result.
            std::vector<void *> domain;                  // global domain, all possible values. init to TOP like in lattice.

            // add all used values to the domain set, initialize to TOP (in LVA, it is set of all variables --> domain)
            for (auto I = inst_begin(F); I != inst_end(F); ++I)
            {
                if (Instruction *inst = dyn_cast<Instruction>(&*I))
                {
                    for (auto operands = inst->op_begin(); operands != inst->op_end(); ++operands)
                    {
                        Value *operand = *operands;

                        // Check operand type isa<X> and type_name as well.
                        if (isa<Instruction>(operand) || isa<Argument>(operand))
                        {
                            if (std::find(domain.begin(), domain.end(), operand) == domain.end())
                            {
                                domain.push_back(operand); // if operand, we push back to domain set.
                            }
                        }
                    }
                }
            }

            // initialize analysis
            LiveVariableAnalysis analysis = LiveVariableAnalysis(domain);
            std::set<int> boudary = std::set<int>(), bbonly = std::set<int>();
            fresult = analysis.pre_analysis(F, boudary, bbonly);

            // errs() << "Function Name : " << F.getName() << "\n";

            // We have got in/out for each block, now we need to analyze each instruction
            for (auto &BB : F)
            {

                errs() << "Basic Block : " << BB.getName() << "\n";

                std::set<int> live = fresult[&BB].out; // out contains the Live Variables list.

                for (auto inst = BB.rbegin(); inst != BB.rend(); ++inst)
                { // backward analysis.
                    Instruction *I = &*inst;

                    // PHINode is not a real node, so no need to add liveness behind it
                    if (auto phi = dyn_cast<PHINode>(&*inst))
                    {
                        int i = analysis.domainIndex(phi); // index in domain set.
                        auto LHS = live.find(i);
                        // if something has been redefined, kill it
                        if (LHS != live.end())
                        {
                            // found live, kill it.
                            live.erase(LHS);
                        }
                        // Phi node, no live set.
                        auto s = std::set<std::string>();
                        output.push_back(std::make_pair(I, s));
                    }
                    else
                    {
                        for (auto op = inst->op_begin(); op != inst->op_end(); ++op)
                        {
                            Value *val = *op;

                            // find live varaible in set and add it.
                            if (isa<Instruction>(val) || isa<Argument>(val))
                            {
                                int i = analysis.domainIndex(val); // operand found live here.
                                if (i != analysis.INDEX_NOT_FOUND)
                                {
                                    live.insert(i); // insert, (duplicates are automatically rejected at insert(). )
                                }
                            }
                        }

                        // kill the redefined varaible
                        int i = analysis.domainIndex(&*inst);
                        auto iter = live.find(i);
                        if (iter != live.end())
                        {
                            live.erase(iter);
                        }

                        std::set<std::string> sset;
                        int count = 0;
                        for (auto val : live)
                        {
                            auto str = getValueName((Value *)analysis.domain[val]);
                            sset.insert(str);
                        }

                        liveSetElems.push_back(sset);
                        output.push_back(std::make_pair(I, sset));
                    }
                }

                for (auto o = output.rbegin(); o != output.rend(); ++o)
                {
                    histogram[o->second.size()]++;
                    errs() << "Program Point " << counter++ << ", instr :" << *o->first << ", live_set = ";
                    print_container(o->second);
                    errs() << "\n";
                }

                errs() << "\nFor Basic Block " << BB.getName() << "; OUT_SET = ";
                print_container((output.begin())->second);
                errs() << "\n\n";
            }

            if (iteration > 1)
            {

                errs() << "Histogram : \n";
                errs() << "#live_count , #program_points_count  \n\n";

                for (auto &elem : histogram)
                {
                    errs() << elem.first << ", " << elem.second << "\n";
                }
            }

            iteration++;
            return false;
        }
    };
} // namespace

char Liveness::ID = 0;
static RegisterPass<Liveness> X("lva", "Live Variable Analysis Pass", true, true);

static RegisterStandardPasses Y(
    PassManagerBuilder::EP_EarlyAsPossible,
    [](const PassManagerBuilder &Builder,
       legacy::PassManagerBase &PM) { PM.add(new Liveness()); });
#include "llvm/Pass.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <queue>

/**
 * References : https://github.com/jarulraj/llvm/, https://llvm.org/doxygen/LiveVariables_8cpp_source.html
 * This file contains all the ADTs and struct definitions needed for Live Variable Analysis. 
*/

namespace llvm
{
	void error(std::string);

	// Expression for storing BinaryInstruction, easier for comparasion
	class Expression
	{
	public:
		Value *v1, *v2;
		Instruction::BinaryOps op;	  // SSA form binary operand ins considered.
		Expression(Instruction *ins); // String equivalent expression from SSA IR.
		bool operator==(const Expression &exp);
	};

	// union two set s1 U s2.
	template <typename T>
	std::set<T> union_set(std::set<T> s1, std::set<T> s2)
	{
		std::set<T> result;
		result.insert(s1.begin(), s1.end());
		result.insert(s2.begin(), s2.end());
		return result;
	}

	// intersection of two sets s1 <> s2.
	template <typename T>
	std::set<T> intersect_set(std::set<T> s1, std::set<T> s2)
	{
		std::set<T> result;
		for (auto i = s1.begin(); i != s1.end(); i++)
		{
			auto iter = s2.find(*i);
			if (iter != s2.end())
			{
				result.insert(*i);
			}
		}
		return result;
	}

	// set difference of two sets s1 - s2.
	template <typename T>
	std::set<T> diff_set(std::set<T> s1, std::set<T> s2)
	{
		std::set<T> result;
		result.insert(s1.begin(), s1.end());
		for (auto i = s2.begin(); i != s2.end(); ++i)
		{
			auto iter = result.find(*i);
			if (iter != result.end())
			{
				result.erase(iter);
			}
		}
		return result;
	}

	// result for transfer function
	struct BlockTransferFunction
	{
		std::set<int> transfer;
		std::map<BasicBlock *, std::set<int>> neighbor;
	};

	// in and out results for blocks, @BB Level.
	struct BlockResult
	{
		std::set<int> in, out;
		BlockTransferFunction transferOutput;
	};

	// generic backward with union dataflow framework.
	class Dataflow
	{
	public:
		Dataflow(std::vector<void *> domain)
			: domain(domain){};
		std::set<int> MeetOperationSets(std::vector<std::set<int>> input);
		std::map<BasicBlock *, BlockResult> pre_analysis(Function &F, std::set<int> boudary, std::set<int> interior);
		int domainIndex(void *ptr);
		virtual BlockTransferFunction transferFn(std::set<int> input, BasicBlock *currentBlock) = 0;
		const int INDEX_NOT_FOUND = -1;
		std::vector<void *> domain;
	};
}; // namespace llvm
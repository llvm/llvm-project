#include "ICFG.h"
#include "utils.h"

int ICFG::getNodeId(int fid, int bid) {
    auto it = nodeIdOfFunctionBlock.find({fid, bid});
    if (it != nodeIdOfFunctionBlock.end())
        return it->second;
    int id = nodeIdOfFunctionBlock.size();
    nodeIdOfFunctionBlock[{fid, bid}] = id;
    functionBlockOfNodeId.push_back({fid, bid});
    return id;
}

void ICFG::addNormalEdge(int fid, int uBid, int vBid) {
    int u = getNodeId(fid, uBid);
    int v = getNodeId(fid, vBid);
    // llvm::errs() << "In fun: " << Global.functionLocations[fid]->name << " "
    //              << uBid << " -> " << vBid << "\n";
    G[u].push_back({Edge::Type::INTRA_PROC, 0, v});
}

void ICFG::addCallAndReturnEdge(int uEntry, int uExit, int calleeFid) {
    // entry & exit nodes of callee
    auto [entry, exit] = entryExitOfFunction[calleeFid];
    int vEntry = getNodeId(calleeFid, entry);
    int vExit = getNodeId(calleeFid, exit);

    callSiteId++;
    // llvm::errs()
    //     << "Call: "
    //     <<
    //     Global.functionLocations[functionBlockOfNodeId[uEntry].first]->name
    //     << " -> " << Global.functionLocations[calleeFid]->name << "\n";

    G[uEntry].push_back({Edge::Type::CALL_EDGE, callSiteId, vEntry});
    G[vExit].push_back({Edge::Type::RETURN_EDGE, callSiteId, uExit});
}

void ICFG::tryAndAddCallSite(int fid, const CFGBlock &B) {
    if (B.empty()) // block has no element
        return;

    std::optional<CFGStmt> CS = B.back().getAs<CFGStmt>();
    if (!CS) // the last element is not a CFGBlock
        return;

    const CallExpr *expr = dyn_cast<CallExpr>(CS->getStmt());
    if (!expr) // the last stmt is not a CallExpr
        return;

    const FunctionDecl *calleeDecl = expr->getDirectCallee();
    if (!calleeDecl)
        return;

    std::string callee = getFullSignature(calleeDecl);
    int calleeFid = Global.getIdOfFunction(callee);
    if (calleeFid == -1)
        return;

    // callsite block has only one successor
    assert(B.succ_size() == 1);
    const CFGBlock &Succ = **B.succ_begin();

    int uEntry = getNodeId(fid, B.getBlockID());
    int uExit = getNodeId(fid, Succ.getBlockID());

    if (!visited(calleeFid)) {
        // callee has yet to be added to this ICFG
        callsitesToProcess[calleeFid].push_back({uEntry, uExit});
    } else {
        addCallAndReturnEdge(uEntry, uExit, calleeFid);
    }
}

void ICFG::addFunction(int fid, const CFG &cfg) {
    if (fid == -1 || visited(fid))
        return;

    n += cfg.size();
    G.resize(n);

    entryExitOfFunction[fid] = {cfg.getEntry().getBlockID(),
                                cfg.getExit().getBlockID()};

    // if (cfg.size() <= 3)
    //     return;

    // add all blocks to graph
    // 先把所有的 CFGBlock 都加入到图中，这样顺序好点
    for (auto BI = cfg.begin(); BI != cfg.end(); ++BI) {
        const CFGBlock &B = **BI;
        getNodeId(fid, B.getBlockID());
    }

    // traverse each block
    for (auto BI = cfg.begin(); BI != cfg.end(); ++BI) {
        const CFGBlock &B = **BI;

        // traverse all successors to add normal edges
        for (auto SI = B.succ_begin(); SI != B.succ_end(); ++SI) {
            const CFGBlock *Succ = *SI;
            // Successor may be null in case of optimized-out edges. See:
            // https://clang.llvm.org/doxygen/classclang_1_1CFGBlock.html#details
            if (!Succ)
                continue;
            addNormalEdge(fid, B.getBlockID(), Succ->getBlockID());
        }

        tryAndAddCallSite(fid, B);
    }

    // process callsites
    for (const auto &[uEntry, uExit] : callsitesToProcess[fid]) {
        addCallAndReturnEdge(uEntry, uExit, fid);
    }
    callsitesToProcess.erase(fid);

    // cfg.viewCFG(LangOptions());
}
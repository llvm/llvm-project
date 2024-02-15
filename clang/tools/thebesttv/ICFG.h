#pragma once

#include "clang/Analysis/CFG.h"

using namespace clang;

struct ICFG {
    struct Edge {
        enum class Type {
            INTRA_PROC, // 过程内的，目前不区分 call to return edge 和普通的边
            CALL_EDGE, // 同一个调用点，call edge 和 return edge 的 id 必须匹配
            RETURN_EDGE
        };
        Type type;
        int callSiteId;
        int target;
    };
    std::vector<std::vector<Edge>> G;
    // 过程内的边 id 默认为 0，call edge 和 return edge 的 id 从 1 开始，
    // 也就是调用点的 id。
    int callSiteId = 0;

    std::map<int, std::pair<int, int>> entryExitOfFunction;

    int n = 0;
    // <function id, CFGBlock id> -> node id (of current graph);
    std::map<std::pair<int, int>, int> nodeIdOfFunctionBlock;
    std::vector<std::pair<int, int>> functionBlockOfNodeId;

    int getNodeId(int fid, int bid);

    void addNormalEdge(int fid, int uBid, int vBid);

    // callee -> all callsites (each callsite is a node id)
    std::map<int, std::vector<std::pair<int, int>>> callsitesToProcess;

    void addCallAndReturnEdge(int uEntry, int uExit, int calleeFid);

    /**
     * 尝试处理调用点，并添加 call & return edge
     *
     * 如果 B 的最后一条语句是调用点，而且可以找到 callee：
     * 1. 如果 callee 还没有被添加到 ICFG 中，那么将 callsite
     *    记录下来，之后再处理
     * 2. 如果 callee 已经被添加到 ICFG 中，那么直接添加 call & return edge
     */
    void tryAndAddCallSite(int fid, const CFGBlock &B);

    bool visited(int fid) {
        return entryExitOfFunction.find(fid) != entryExitOfFunction.end();
    }

    void addFunction(int fid, const CFG &cfg);
};

#pragma once

#include "utils.h"
#include <random>

struct ICFGPathFinder {
    const ICFG &icfg;
    ICFGPathFinder(const ICFG &icfg) : icfg(icfg) {}

    std::set<std::vector<int>> results;

    virtual void search(int source, int target,
                        const std::vector<int> &pointsToPass,
                        const std::vector<int> &pointsToAvoid,
                        int maxCallDepth) = 0;
};

struct DfsPathFinder : public ICFGPathFinder {
    DfsPathFinder(const ICFG &icfg) : ICFGPathFinder(icfg) {}

    int source;
    int target;
    int maxCallDepth;
    std::vector<int> path;
    std::vector<bool> visiting;

    std::stack<int> callStack; // 部分平衡的括号匹配
    std::set<int> callSites;

    /**
     * TODO: 目前 pointsToAvoid 还没处理
     */
    void search(int source, int target, const std::vector<int> &pointsToPass,
                const std::vector<int> &pointsToAvoid,
                int maxCallDepth) override {
        this->source = source;
        this->target = target;
        this->maxCallDepth = maxCallDepth;

        logger.info("=== DfsPathFinder ===");
        logger.info("source: {}", source);
        logger.info("target: {}", target);
        logger.info("maxCallDepth: {}", maxCallDepth);
        logger.info("pass:   {}", fmt::join(pointsToPass, " "));
        logger.info("avoid:  {}", fmt::join(pointsToAvoid, " "));

        path.clear();
        path.push_back(source);

        visiting.resize(icfg.n);
        std::fill(visiting.begin(), visiting.end(), false);

        while (!callStack.empty())
            callStack.pop();
        callSites.clear();

        dfs(source);

        // filter: mark paths that do not contain pathFilter
        std::vector<std::set<std::vector<int>>::iterator> toDelete;
        for (auto it = results.begin(); it != results.end(); ++it) {
            const auto &path = *it;
            bool match = true;
            int i = 0;
            for (int x : path) {
                if (i >= pointsToPass.size())
                    break;
                if (x == pointsToPass[i]) {
                    i++;
                }
            }
            if (i != pointsToPass.size()) {
                toDelete.push_back(it);
            }
        }

        for (auto it : toDelete) {
            results.erase(it);
        }
    }

    void dfs(int u) {
        if (results.size() > 40)
            return;
        if (u == target) {
            logger.info("Found path: {}", fmt::join(path, " "));
            results.insert(path);
            return;
        }

        // std::vector<ICFG::Edge> intraEdges, interEdges;
        // for (const auto &e : icfg.G[u]) {
        //     if (e.type == ICFG::Edge::Type::INTRA_PROC) {
        //         intraEdges.push_back(e);
        //     } else {
        //         interEdges.push_back(e);
        //     }
        // }
        // // 选择所有过程间的边，再随机选一条过程内的边
        // std::vector<ICFG::Edge> edgesToVisit = interEdges;
        // // https://en.cppreference.com/w/cpp/algorithm/sample
        // std::sample(intraEdges.begin(), intraEdges.end(),
        //             std::back_inserter(edgesToVisit), 1,
        //             std::mt19937{std::random_device{}()});
        // logger.info("Size: {}", edgesToVisit.size());

        // for (const auto &e : edgesToVisit) {
        for (const auto &e : icfg.G[u]) {
            int v = e.target;
            if (visiting[v])
                continue;

            if (e.type == ICFG::Edge::Type::INTRA_PROC) {
                visiting[v] = true;
                path.push_back(v);
                dfs(v);
                path.pop_back();
                visiting[v] = false;
            } else {
                std::stack<int> oldCallStack = callStack;
                std::set<int> oldCallSites = callSites;

                if (e.type == ICFG::Edge::Type::CALL_EDGE) { // 左括号
                    callStack.push(e.callSiteId);
                } else { // 右括号
                    if (!callStack.empty()) {
                        if (callStack.top() != e.callSiteId) {
                            continue;
                        } else {
                            callStack.pop();
                        }
                    }
                }
                callSites.insert(e.callSiteId);

                if (callSites.size() <= maxCallDepth) {
                    visiting[v] = true;
                    path.push_back(v);
                    dfs(v);
                    path.pop_back();
                    visiting[v] = false;
                }

                callSites = oldCallSites;
                callStack = oldCallStack;
            }
        }
    }
};

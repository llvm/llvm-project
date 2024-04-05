#pragma once

#include "utils.h"
#include <random>

struct ICFGPathFinder {
    const ICFG &icfg;
    ICFGPathFinder(const ICFG &icfg) : icfg(icfg) {}

    std::set<std::vector<int>> results;

    virtual void search(int source, int target,
                        const std::vector<int> &pointsToPass,
                        const std::set<int> &pointsToAvoid,
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
                const std::set<int> &pointsToAvoid, int maxCallDepth) override {
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

struct DijPathFinder : public ICFGPathFinder {
    const int INF = 0x3f3f3f3f;
    std::vector<int> d, fa;

    DijPathFinder(const ICFG &icfg)
        : ICFGPathFinder(icfg), d(icfg.n), fa(icfg.n) {}

    struct Node {
        int u, d, fa;
        bool operator<(const Node &b) const { return d > b.d; }
    };

    void dij(int s, const std::set<int> &pointsToAvoid) {
        std::fill(d.begin(), d.end(), INF);
        d[s] = 0;

        std::fill(fa.begin(), fa.end(), -1);

        std::priority_queue<Node> q;
        q.push({s, 0, -1});

        while (!q.empty()) {
            Node p = q.top();
            q.pop();
            int u = p.u;
            if (p.d != d[u])
                continue;

            fa[u] = p.fa;

            for (const auto &e : icfg.G[u]) {
                int v = e.target;
                // skip point to avoid
                if (pointsToAvoid.count(v))
                    continue;
                if (d[v] > d[u] + 1) {
                    d[v] = d[u] + 1;
                    q.push({v, d[v], u});
                }
            }
        }
    }

    std::vector<int> trace(int u) {
        std::vector<int> path;
        while (u != -1) {
            path.push_back(u);
            u = fa[u];
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    void removeConsecutiveDuplicates(std::vector<int> &v) {
        if (v.size() <= 1)
            return;
        std::vector<int> res;
        res.push_back(v[0]);
        for (int i = 1; i < v.size(); i++) {
            if (v[i] != v[i - 1]) {
                res.push_back(v[i]);
            }
        }
        v = res;
    }

    void search(int source, int target, const std::vector<int> &pointsToPass,
                const std::set<int> &pointsToAvoid, int maxCallDepth) override {
        if (pointsToAvoid.count(source)) {
            logger.warn("Source is in pointsToAvoid!");
            return;
        }
        if (pointsToAvoid.count(target)) {
            logger.warn("Target is in pointsToAvoid!");
            return;
        }
        for (int x : pointsToPass) {
            if (pointsToAvoid.count(x)) {
                logger.warn("Point to pass is in pointsToAvoid!");
                return;
            }
        }

        if (source == target) {
            logger.info("Source the same as target, skip searching");
            results.insert({source});
            return;
        }

        std::vector<int> ptp;
        ptp.push_back(source);
        for (int x : pointsToPass) {
            ptp.push_back(x);
        }
        ptp.push_back(target);

        logger.info("=== DijPathFinder ===");
        logger.info("Original path to search: {}", fmt::join(ptp, " "));

        removeConsecutiveDuplicates(ptp);

        logger.info("Deduplicated path: {}", fmt::join(ptp, " "));

        std::vector<int> path;
        for (int i = 0; i < ptp.size() - 1; i++) {
            int u = ptp[i], v = ptp[i + 1];
            dij(u, pointsToAvoid);
            if (d[v] == INF) {
                logger.warn("No path from {} to {}", u, v);
                return;
            }
            std::vector<int> p = trace(v);
            p.pop_back();
            path.insert(path.end(), p.begin(), p.end());
            logger.info("Path {} -> {}, d = {}, p: {}", u, v, d[v],
                        fmt::join(p, " "));
        }
        path.push_back(target);

        results.insert(path);
    }
};
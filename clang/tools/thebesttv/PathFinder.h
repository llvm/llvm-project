#pragma once

#include "utils.h"

#include <chrono>
#include <exception>

using time_point = std::chrono::steady_clock::time_point;

struct ICFGPathFinder {
    const ICFG &icfg;
    ICFGPathFinder(const ICFG &icfg) : icfg(icfg) {}

    std::set<std::vector<int>> results;

    void search(int source, int target, const std::vector<int> &pointsToPass,
                const std::set<int> &pointsToAvoid, int maxCallDepth) {
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
                logger.warn("Point to pass ({}) is in pointsToAvoid!", x);
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

        logger.info("=== ICFGPathFinder ===");
        logger.info("Original path to search: {}", fmt::join(ptp, " "));

        removeConsecutiveDuplicates(ptp);

        logger.info("Deduplicated path: {}", fmt::join(ptp, " "));

        _search(ptp, pointsToAvoid, maxCallDepth);
    }

  private:
    virtual void _search(const std::vector<int> &pointsToPass,
                         const std::set<int> &pointsToAvoid,
                         int maxCallDepth) = 0;

    static void removeConsecutiveDuplicates(std::vector<int> &v) {
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
};

struct IntraProceduralBfs {
  private:
    const ICFG &icfg;
    const int fid;
    const std::pair<int, int> nodeRange;
    const int n; // number of nodes
    std::vector<int> visited, fa;

    static std::pair<int, int> getNodeRange(const ICFG &icfg, int fid) {
        auto [entry, exit] = icfg.entryExitOfFunction.at(fid);
        entry = icfg.nodeIdOfFunctionBlock.at({fid, entry});
        exit = icfg.nodeIdOfFunctionBlock.at({fid, exit});
        return {std::min(entry, exit), std::max(entry, exit)};
    }

    bool inFunction(int u) const {
        return nodeRange.first <= u && u <= nodeRange.second;
    }

    int getIndex(int u) const { return u - nodeRange.first; }

    // 在 fid 代表的函数中进行 BFS
    void bfs(int u, const std::set<int> &pointsToAvoid) {
        requireTrue(inFunction(u), "BFS source not in function");

        std::queue<int> q;
        q.push(u);

        std::fill(visited.begin(), visited.end(), false);
        visited[getIndex(u)] = true;
        fa[getIndex(u)] = -1;

        while (!q.empty()) {
            u = q.front();
            q.pop();
            for (const auto &e : icfg.G[u]) {
                int v = e.target;
                int vi = getIndex(v);
                if (pointsToAvoid.count(v))
                    continue;
                if (e.type != ICFG::Edge::Type::INTRA_PROC)
                    continue;
                requireTrue(inFunction(v));
                if (visited[vi])
                    continue;
                q.push(v);
                visited[vi] = true;
                fa[vi] = u;
            }
        }
    }

  public:
    IntraProceduralBfs(const ICFG &icfg, const std::set<int> &pointsToAvoid,
                       int fid, int u)
        : icfg(icfg), fid(fid), nodeRange(getNodeRange(icfg, fid)),
          n(nodeRange.second - nodeRange.first + 1), visited(n), fa(n) {
        bfs(u, pointsToAvoid);
    }

    bool reachable(int v) const {
        if (!inFunction(v))
            return false;
        return visited[getIndex(v)];
    }

    std::vector<int> getAllReachableNodes() const {
        std::vector<int> res;
        for (int i = nodeRange.first; i <= nodeRange.second; i++) {
            if (visited[getIndex(i)]) {
                res.push_back(i);
            }
        }
        return res;
    }

    std::vector<int> trace(int u) const {
        std::vector<int> path;
        while (u != -1) {
            path.push_back(u);
            u = fa[getIndex(u)];
        }
        std::reverse(path.begin(), path.end());
        return path;
    }
};

struct InterProceduralDij {
  private:
    const ICFG &icfg;
    const int INF = 0x3f3f3f3f;
    std::vector<int> d, fa;
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

            for (const auto &e : icfg.G_reverse[u]) {
                int v = e.target;
                // skip point to avoid
                if (pointsToAvoid.count(v))
                    continue;
                // 只考虑调用边和返回边
                int w = (e.type == ICFG::Edge::Type::INTRA_PROC) ? 0 : 1;
                if (d[v] > d[u] + w) {
                    d[v] = d[u] + w;
                    q.push({v, d[v], u});
                }
            }
        }
    }

  public:
    InterProceduralDij(const ICFG &icfg, const std::set<int> &pointsToAvoid,
                       int s)
        : icfg(icfg), d(icfg.n), fa(icfg.n) {
        dij(s, pointsToAvoid);
    }

    std::vector<int> trace(int u) const {
        std::vector<int> path;
        while (u != -1) {
            path.push_back(u);
            u = fa[u];
        }
        std::reverse(path.begin(), path.end());
        return path;
    }

    std::optional<int> distance(int v) const {
        if (d[v] == INF)
            return std::nullopt;
        return d[v];
    }
};

struct DfsPathFinder : public ICFGPathFinder {
    DfsPathFinder(const ICFG &icfg) : ICFGPathFinder(icfg) {}

    int maxCallDepth;
    std::set<int> pointsToAvoid;
    std::vector<int> targets; // reversed( pointsToPass + [target] )

    std::vector<int> path;
    std::stack<int> callStack; // 部分平衡的括号匹配

    time_point dfsStartTime;
    int dfsCounter;

    class PathFoundException : public std::exception {
      public:
        const char *what() const noexcept override { return "Path found"; }
    };

    class TimeLimitExceededException : public std::exception {
      public:
        const char *what() const noexcept override {
            return "Time limit exceeded";
        }
    };

    // {fid, u} -> bfs
    std::map<std::pair<int, int>, const IntraProceduralBfs> bfsMap;
    const IntraProceduralBfs &getBfs(int fid, int u) {
        auto key = std::make_pair(fid, u);
        auto it = bfsMap.find(key);
        if (it == bfsMap.end()) {
            it = bfsMap
                     .emplace(key,
                              IntraProceduralBfs(icfg, pointsToAvoid, fid, u))
                     .first;
        }
        return it->second;
    }

    std::map<int, const InterProceduralDij> dijMap;
    const InterProceduralDij &getDij(int u) {
        auto it = dijMap.find(u);
        if (it == dijMap.end()) {
            it = dijMap.emplace(u, InterProceduralDij(icfg, pointsToAvoid, u))
                     .first;
        }
        return it->second;
    }

    void dfsUpdate() {
        /**
         * 1e6 tick 大约对应 1s
         * 10'000'000:
         * [2024-05-22 10:58:08.854] [info]  tick!
         * [2024-05-22 10:58:20.508] [info]  tick!
         * [2024-05-22 10:58:32.554] [info]  tick!
         * [2024-05-22 10:58:44.367] [info]  tick!
         */
        dfsCounter++;
        if (dfsCounter >= Global.dfsTick) {
            time_point currentTime = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                                currentTime - dfsStartTime)
                                .count();
            logger.info("  DFS taking {}s ...", duration);
            if (duration > Global.dfsTimeout)
                throw TimeLimitExceededException();
            dfsCounter = 0;
        }
    }

  private:
    void _search(const std::vector<int> &pointsToPass,
                 const std::set<int> &pointsToAvoid,
                 int maxCallDepth) override {
        int source = pointsToPass.front();

        this->maxCallDepth = maxCallDepth;
        this->pointsToAvoid = pointsToAvoid;

        this->targets =
            std::vector<int>(pointsToPass.begin() + 1, pointsToPass.end());
        // 为了输出，先不反转

        logger.info("=== DfsPathFinder ===");
        logger.info("source: {}", source);
        logger.info("maxCallDepth: {}", maxCallDepth);
        logger.info("pass:   {}", fmt::join(this->targets, " "));
        logger.info("avoid:  {}", fmt::join(this->pointsToAvoid, " "));

        // 输出完后，把 targets 反转
        std::reverse(this->targets.begin(), this->targets.end());

        try {
            dfsStartTime = std::chrono::steady_clock::now();
            dfsCounter = 0;
            dfs(source, 0);
        } catch (const TimeLimitExceededException &e) {
            logger.warn("Time limit exceeded, DFS took too long");
        } catch (const PathFoundException &e) {
            // path found
            std::vector<int> fullPath;
            fullPath.push_back(source);
            for (auto it = path.begin() + 1; it != path.end(); it++) {
                int u = fullPath.back();
                int v = *it;

                const int uFid = Global.icfg.functionBlockOfNodeId[u].first;
                const int vFid = Global.icfg.functionBlockOfNodeId[v].first;

                if (uFid == vFid) {
                    const auto &bfs = getBfs(uFid, u);
                    requireTrue(bfs.reachable(v));
                    std::vector<int> p = bfs.trace(v);
                    fullPath.insert(fullPath.end(), p.begin() + 1, p.end());
                } else {
                    bool found = false;
                    for (const auto &e : icfg.G[u]) {
                        if (e.target == v) {
                            requireTrue(e.type != ICFG::Edge::Type::INTRA_PROC);
                            fullPath.push_back(e.target);
                            found = true;
                            break;
                        }
                    }
                    requireTrue(found);
                }
            }
            results.insert(fullPath);
        }
    }

    void dfs(const int u, int callDepth) {
        dfsUpdate();

        path.push_back(u);

        bool foundOneTarget = (u == targets.back());
        if (foundOneTarget) {
            targets.pop_back();
            if (targets.empty())
                throw PathFoundException();
        }

        const int target = targets.back();
        requireTrue(u != target);

        const int currentFid = Global.icfg.functionBlockOfNodeId[u].first;
        const int targetFid = Global.icfg.functionBlockOfNodeId[target].first;

        const auto &bfs = getBfs(currentFid, u);

        if (currentFid == targetFid) {
            // 过程内
            if (bfs.reachable(target)) {
                dfs(target, callDepth);
            }
        } else if (callDepth < maxCallDepth) {
            /**
             * 过程间
             * 做两遍最短路（可达性 + 反图最短路）
             *
             * 1. 过程内可达性，找到 u 可以到达的所有过程内节点
             *    并且，这些节点有 调用边 或 返回边，可以一步走到其他函数。
             *
             * 2. 过程间最短路，只考虑调用边和返回边的权重，
             *    求来自 1 的那些过程内节点，到 target 的最短路。
             *    对应 ICFG 反图上的 Dij
             */

            const auto &dij = getDij(target);

            struct Node {
                int v;           // u 可以到达的过程内节点
                ICFG::Edge edge; // 过程间边（调用边或返回边）
                int d; // 从 edge.target 到 target 的距离，从而强制 v
                       // 走过程间边另一个函数

                Node(int v, ICFG::Edge edge, int d) : v(v), edge(edge), d(d) {}

                bool operator<(const Node &b) const { return d < b.d; }
            };

            std::vector<Node> nodes;
            for (int v : bfs.getAllReachableNodes()) {
                for (const auto &e : icfg.G[v]) {
                    if (e.type == ICFG::Edge::Type::INTRA_PROC)
                        continue;

                    // 右括号
                    if (e.type == ICFG::Edge::Type::RETURN_EDGE &&
                        // 对应的左括号不匹配
                        !callStack.empty() && callStack.top() != e.callSiteId) {
                        continue;
                    }

                    auto d = dij.distance(e.target);
                    if (d.has_value()) {
                        nodes.emplace_back(v, e, d.value());
                    }
                }
            }

            std::sort(nodes.begin(), nodes.end());
            for (const auto &node : nodes) {
                // 从 u -> node.v -(1)-> node.edge.target -(node.d)-> target
                // 经过的调用深度超过，就直接跳过
                if (callDepth + node.d + 1 > maxCallDepth)
                    break;

                const auto &e = node.edge;

                std::stack<int> oldCallStack = callStack;

                if (e.type == ICFG::Edge::Type::CALL_EDGE) { // 左括号
                    callStack.push(e.callSiteId);
                } else { // 右括号
                    if (!callStack.empty()) {
                        requireTrue(callStack.top() == e.callSiteId);
                        callStack.pop();
                    }
                }

                if (node.v == u) {
                    // node.v 和 u 相同时，意味着可以直接从 u 出发，
                    // 走一步 node.e 到达新的函数
                    dfs(node.edge.target, callDepth + 1);
                } else {
                    // 需要先过程内从 u 走到 node.v
                    // 然后再走一步 node.e 到达新的函数
                    path.push_back(node.v);
                    dfs(node.edge.target, callDepth + 1);
                    path.pop_back();
                }

                callStack = oldCallStack;
            }
        }

        if (foundOneTarget) {
            targets.push_back(u);
        }

        path.pop_back();
    }
};

struct DijPathFinder : public ICFGPathFinder {
    const int INF = 0x3f3f3f3f;
    std::vector<int> d, fa;

    DijPathFinder(const ICFG &icfg)
        : ICFGPathFinder(icfg), d(icfg.n), fa(icfg.n) {}

  private:
    struct Node {
        int u, d, fa;
        bool operator<(const Node &b) const { return d > b.d; }
    };

    void dij(int s, int target, const std::set<int> &pointsToAvoid) {
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

            // 找到 s -> target 的最短后就退出
            if (u == target)
                break;

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

    void _search(const std::vector<int> &ptp,
                 const std::set<int> &pointsToAvoid,
                 int maxCallDepth) override {
        int target = ptp.back();

        std::vector<int> path;
        for (int i = 0; i < ptp.size() - 1; i++) {
            int u = ptp[i], v = ptp[i + 1];
            dij(u, v, pointsToAvoid);
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
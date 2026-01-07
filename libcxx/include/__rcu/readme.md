# folly


## domain

- `lock` :
  - increment the counter with the epoch number == `_version`

- `unlock`:
  - decrement the counter (without arguments)

- `retire` :
  - push `cb` to `_q`
  - if long time since last sync, try try_lock sync mutex and `half_sync` without blocking (to get some finished nodes)
    - put the finished `cb`s to `executor` (immideately execute)

- `half_sync` :
  - `current` is the current `_version` number, `next` is +1
  - move all nodes from `_q` to `queues[0]` (cannot just have a single `queue` and swap because of later readers) (thread safe with concurrent more push to `_q`)
  - if is blocking, wait for zero for the epoch `next & 1`
  - else, if epoch `next & 1` has reader is true, return
  - at this stage, `next & 1` epoch reader zero, if late reader comes, it increments the `current` reader count. (no concurrent half_sync as it is mutex protected)
  - move all nodes from `queue[1]` to `finished`, and move all nodes from `queue[0]` to `queue[1]`
  - store `next` to `_version`
  - notify threads that turn_.waitForTurn(next)

- `sync`
  - `current` is the current `_version` number, `target` is +2
  - while true
     - if current `work_` is smaller than `target` and cas to `target` succeeded,
        `half_sync` until `version_` >= `target`, and run all the finished `cb`s, `return`
     - else 
       - if `version_` >= `target` , `return`
       - else `turn_.waitForTurn(work)` (so if other's target >= our target, other's second epoch half sync will unblock us)



## executor

- immediately invoke
- queue if the `f` schedule another `f2` in the `executor`


## example

version == 0

  T1                            T2                       T3

Reader1 lock (0, 1)
read obj0
                       obj = obj1
                       retire(obj0)
                       sync, target = 2
                       half, next = 1
                       wait_zero(epoch == 1)
                       cb0 -> queue[1]
                       version = 1
                       notify(1)
                                                   Read2 lock (1, 1)
                                                   read obj1
                      half, next = 2
unlock (0,0)
                      wait_zero(epoch==0)
                      cb0 -> finished
                      run cb0
                      version = 2
                      notify(2)
                                                  unlock

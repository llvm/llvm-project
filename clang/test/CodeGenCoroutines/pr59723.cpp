// This is reduced test case from https://github.com/llvm/llvm-project/issues/59723.
// This is not a minimal reproducer intentionally to check the compiler's ability.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fcxx-exceptions\
// RUN:     -fexceptions -O2 -emit-llvm %s -o - | FileCheck %s

#include "Inputs/coroutine.h"

// executor and operation base

class bug_any_executor;

struct bug_async_op_base
{
	void invoke();

protected:

	~bug_async_op_base() = default;
};

class bug_any_executor
{
	using op_type = bug_async_op_base;

public:

	virtual ~bug_any_executor() = default;

	// removing noexcept enables clang to find that the pointer has escaped
	virtual void post(op_type& op) noexcept = 0;

	virtual void wait() noexcept = 0;
};

class bug_thread_executor : public bug_any_executor
{

public:

	void start()
	{
		
	}

	~bug_thread_executor()
	{
	}

	// although this implementation is not realy noexcept due to allocation but I have a real one that is and required to be noexcept
	virtual void post(bug_async_op_base& op) noexcept override;

	virtual void wait() noexcept override
	{
		
	}
};

// task and promise

struct bug_final_suspend_notification
{
	virtual std::coroutine_handle<> get_waiter() = 0;
};

class bug_task;

class bug_task_promise
{
	friend bug_task;
public:

	bug_task get_return_object() noexcept;

	constexpr std::suspend_always initial_suspend() noexcept { return {}; }

	std::suspend_always final_suspend() noexcept 
	{
		return {};
	}

	void unhandled_exception() noexcept;

	constexpr void return_void() const noexcept {}

	void get_result() const
	{
		
	}
};

template <class T, class U>
T exchange(T &&t, U &&u) {
    T ret = t;
    t = u;
    return ret;
}

class bug_task
{
	friend bug_task_promise;
	using handle = std::coroutine_handle<>;
	using promise_t = bug_task_promise;

	bug_task(handle coro, promise_t* p) noexcept : this_coro{ coro }, this_promise{ p }
	{
	
	}

public:
	using promise_type = bug_task_promise;

    bug_task(bug_task&& other) noexcept
		: this_coro{ exchange(other.this_coro, nullptr) }, this_promise{ exchange(other.this_promise, nullptr) } { 
		
	}

	~bug_task()
	{
		if (this_coro)
			this_coro.destroy();
	}

	constexpr bool await_ready() const noexcept
	{
		return false;
	}

	handle await_suspend(handle waiter) noexcept
	{
		return this_coro;
	}

	void await_resume() 
	{
		return this_promise->get_result();
	}

	handle this_coro;
	promise_t* this_promise;
};

bug_task bug_task_promise::get_return_object() noexcept
{
	return { std::coroutine_handle<bug_task_promise>::from_promise(*this), this };
}

// spawn operation and spawner

template<class Handler>
class bug_spawn_op final : public bug_async_op_base, bug_final_suspend_notification
{
	Handler handler;
	bug_task task_;

public:

	bug_spawn_op(Handler handler, bug_task&& t)
		: handler { handler }, task_{ static_cast<bug_task&&>(t) } {}

	virtual std::coroutine_handle<> get_waiter() override
	{
		handler();
		return std::noop_coroutine();
	}
};

class bug_spawner;

struct bug_spawner_awaiter
{
	bug_spawner& s;
	std::coroutine_handle<> waiter;

	bug_spawner_awaiter(bug_spawner& s) : s{ s } {}

	bool await_ready() const noexcept;

	void await_suspend(std::coroutine_handle<> coro);

	void await_resume() {}
};

class bug_spawner
{
	friend bug_spawner_awaiter;

	struct final_handler_t
	{
		bug_spawner& s;

		void operator()()
		{
			s.awaiter_->waiter.resume();
		}
	};

public:

	bug_spawner(bug_any_executor& ex) : ex_{ ex } {}

	void spawn(bug_task&& t) {
		using op_t = bug_spawn_op<final_handler_t>;
		// move task into ptr
		op_t* ptr = new op_t(final_handler_t{ *this }, static_cast<bug_task&&>(t));
		++count_;
		ex_.post(*ptr); // ptr escapes here thus task escapes but clang can't deduce that unless post() is not noexcept
	}

	bug_spawner_awaiter wait() noexcept { return { *this }; }

private:
	bug_any_executor& ex_; // if bug_thread_executor& is used instead enables clang to detect the escape of the promise
	bug_spawner_awaiter* awaiter_ = nullptr;
	unsigned count_ = 0;
};

// test case

bug_task bug_spawned_task(int id, int inc)
{
	co_return;
}

struct A {
    A();
};

void throwing_fn(bug_spawner& s) {
	s.spawn(bug_spawned_task(1, 2));
    throw A{};
}

// Check that the coroutine frame of bug_spawned_task are allocated from operator new.
// CHECK: define{{.*}}@_Z11throwing_fnR11bug_spawner
// CHECK-NOT: alloc
// CHECK: %[[CALL:.+]] = {{.*}}@_Znwm(i64{{.*}} 24)
// CHECK: store ptr @_Z16bug_spawned_taskii.resume, ptr %[[CALL]]

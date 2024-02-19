# 编译器
CC = g++
# 编译选项
CFLAGS = -Wall -Wextra -std=c++11
# 目标文件
TARGET = npe

# 获取所有的.cpp文件
SRCS = $(wildcard *.cpp)
# 生成所有的目标文件.o
OBJS = $(SRCS:.cpp=.o)

# 默认目标，编译可执行文件
all: $(TARGET)

# 生成可执行文件
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@

# 生成目标文件
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# 清理生成的文件
clean:
	rm -rf $(OBJS) $(TARGET)

# 伪目标，表示clean不是一个文件
.PHONY: clean

对于 php 项目的函数
`static void zend_compile_stmt(zend_ast *ast);`
函数声明和定义在同一个文件中：
`~/php/src/php-8.3.3/Zend/zend_compile.c`
原先的代码会定位到函数声明，38ae53234933 修复后可以定位到函数定义。

其次，`input.json` 中的
``` json
                {
                    "type": "stmt",
                    "file": "/home/thebesttv/vul/llvm-project/graph-generation/vul-parser-benchmark/src/php/src/php-8.3.3/Zend/zend_hash.c",
                    "line": 2677,
                    "column": 1
                },
```
定位到函数定义开头，无法匹配到语句。
导致 `source` 的 `nextFid` 为 -1。
a574cb7f4c38 改成一直寻找下一个有效的语句。

此外，函数调用语句过于复杂
``` cpp
zval *zv = zend_hash_find_known_hash(compile_time ? CG(function_table) : EG(function_table), lcname);
```
6896ebef0db8 改为查询所有匹配语句，看有没有 `CallExpr`。

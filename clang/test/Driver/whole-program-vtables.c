// RUN: %clang -target x86_64-unknown-linux -fwhole-program-vtables -### %s 2>&1 | FileCheck --check-prefix=NO-LTO %s
// NO-LTO: invalid argument '-fwhole-program-vtables' only allowed with '-flto'

// RUN: %clang -target x86_64-unknown-linux -resource-dir=%S/Inputs/resource_dir -flto -fwhole-program-vtables -### -c %s 2>&1 | FileCheck --check-prefix=BLACKLIST %s
// BLACKLIST: "-fwhole-program-vtables-blacklist={{.*}}vtables_blacklist.txt"

// RUN: %clang -target x86_64-unknown-linux -fwhole-program-vtables-blacklist=nonexistent.txt -flto -fwhole-program-vtables -### -c %s 2>&1 | FileCheck --check-prefix=NON-EXISTENT-BLACKLIST %s
// NON-EXISTENT-BLACKLIST: no such file or directory: 'nonexistent.txt'

// RUN: %clang -target x86_64-unknown-linux -fwhole-program-vtables-blacklist=%S/Inputs/resource_dir/vtables_blacklist.txt -flto -fwhole-program-vtables -### -c %s 2>&1 | FileCheck --check-prefix=CUSTOM-BLACKLIST %s
// CUSTOM-BLACKLIST: "-fwhole-program-vtables-blacklist={{.*}}Inputs/resource_dir/vtables_blacklist.txt"

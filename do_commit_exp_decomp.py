import subprocess

def run(cmd):
    p = subprocess.run(cmd, cwd='c:/work/VM/llvm-project', capture_output=True, text=True, check=True)
    return p.stdout.strip()

run(['git', 'add', '-A'])
tree = run(['git', 'write-tree'])
print('Tree:', tree)

parent = run(['git', 'rev-parse', 'HEAD'])
print('Parent:', parent)

author_name = run(['git', 'config', 'user.name'])
author_email = run(['git', 'config', 'user.email'])
print('Author:', author_name, author_email)

msg = 'adds frexpf80, ilogbf80, llogbf80, logbf80'

p = subprocess.run(
    ['git', 'commit-tree', tree, '-p', parent, '-m', msg],
    cwd='c:/work/VM/llvm-project',
    capture_output=True,
    text=True,
    check=True
)
commit = p.stdout.strip()
print('Commit:', commit)

run(['git', 'update-ref', 'refs/heads/users/sukumarsawant/f80/exp_decomp', commit])
print('Branch users/sukumarsawant/f80/exp_decomp updated successfully!')

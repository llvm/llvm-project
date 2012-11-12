from clang.cindex import CompilationDatabase
from clang.cindex import CompilationDatabaseError
from clang.cindex import CompileCommands
from clang.cindex import CompileCommand
import os
import gc

kInputsDir = os.path.join(os.path.dirname(__file__), 'INPUTS')

def test_create_fail():
    """Check we fail loading a database with an assertion"""
    path = os.path.dirname(__file__)
    try:
      cdb = CompilationDatabase.fromDirectory(path)
    except CompilationDatabaseError as e:
      assert e.cdb_error == CompilationDatabaseError.ERROR_CANNOTLOADDATABASE
    else:
      assert False

def test_create():
    """Check we can load a compilation database"""
    cdb = CompilationDatabase.fromDirectory(kInputsDir)

def test_lookup_fail():
    """Check file lookup failure"""
    cdb = CompilationDatabase.fromDirectory(kInputsDir)
    assert cdb.getCompileCommands('file_do_not_exist.cpp') == None

def test_lookup_succeed():
    """Check we get some results if the file exists in the db"""
    cdb = CompilationDatabase.fromDirectory(kInputsDir)
    cmds = cdb.getCompileCommands('/home/john.doe/MyProject/project.cpp')
    assert len(cmds) != 0

def test_1_compilecommand():
    """Check file with single compile command"""
    cdb = CompilationDatabase.fromDirectory(kInputsDir)
    cmds = cdb.getCompileCommands('/home/john.doe/MyProject/project.cpp')
    assert len(cmds) == 1
    assert cmds[0].directory == '/home/john.doe/MyProject'
    expected = [ 'clang++', '-o', 'project.o', '-c',
                 '/home/john.doe/MyProject/project.cpp']
    for arg, exp in zip(cmds[0].arguments, expected):
        assert arg == exp

def test_2_compilecommand():
    """Check file with 2 compile commands"""
    cdb = CompilationDatabase.fromDirectory(kInputsDir)
    cmds = cdb.getCompileCommands('/home/john.doe/MyProject/project2.cpp')
    assert len(cmds) == 2
    expected = [
        { 'wd': '/home/john.doe/MyProjectA',
          'line': ['clang++', '-o', 'project2.o', '-c',
                   '/home/john.doe/MyProject/project2.cpp']},
        { 'wd': '/home/john.doe/MyProjectB',
          'line': ['clang++', '-DFEATURE=1', '-o', 'project2-feature.o', '-c',
                   '/home/john.doe/MyProject/project2.cpp']}
        ]
    for i in range(len(cmds)):
        assert cmds[i].directory == expected[i]['wd']
        for arg, exp in zip(cmds[i].arguments, expected[i]['line']):
            assert arg == exp

def test_compilecommand_iterator_stops():
    """Check that iterator stops after the correct number of elements"""
    cdb = CompilationDatabase.fromDirectory(kInputsDir)
    count = 0
    for cmd in cdb.getCompileCommands('/home/john.doe/MyProject/project2.cpp'):
        count += 1
        assert count <= 2

def test_compilationDB_references():
    """Ensure CompilationsCommands are independent of the database"""
    cdb = CompilationDatabase.fromDirectory(kInputsDir)
    cmds = cdb.getCompileCommands('/home/john.doe/MyProject/project.cpp')
    del cdb
    gc.collect()
    workingdir = cmds[0].directory

def test_compilationCommands_references():
    """Ensure CompilationsCommand keeps a reference to CompilationCommands"""
    cdb = CompilationDatabase.fromDirectory(kInputsDir)
    cmds = cdb.getCompileCommands('/home/john.doe/MyProject/project.cpp')
    del cdb
    cmd0 = cmds[0]
    del cmds
    gc.collect()
    workingdir = cmd0.directory


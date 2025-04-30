# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

########## Make rule for 22406 ##########
FILE = fs22406
SRC2=$(SRC)
EXT = f90
F90= $(FC)

build: $(SRC2)/$(FILE).$(EXT)
	@echo ----------------------------------------- building test $@
	$(FC) $(SRC2)/$(FILE).$(EXT) $(EXTRA_LDFLAGS) -o ./$(FILE).$(EXE)

run:
	@echo ------------------------------------------ nothing to run
	./$(FILE).$(EXE)

verify:

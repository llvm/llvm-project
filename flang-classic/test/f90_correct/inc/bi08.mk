# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

########## Make rule for test bi08  ########

bi08: run
	
build:  $(SRC)/bi08.f90 $(SRC)/bi08c.c
	-$(RM) bi08.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/bi08c.c -o bi08c.$(OBJX)
	-$(FC) -c $(FFLAGS) $(SRC)/bi08.f90 -o bi08.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) bi08.$(OBJX) bi08c.$(OBJX) $(LIBS) -o bi08.$(EXESUFFIX)

run:
	@echo ------------------------------------ executing test bi08
	bi08.$(EXESUFFIX)

verify: ;

bi08.run: run


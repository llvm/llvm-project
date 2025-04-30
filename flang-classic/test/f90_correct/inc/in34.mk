#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test in34  ########


in34: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/in34.f90 fcheck.$(OBJX)
	-$(RM) in34.$(EXESUFFIX) in34.$(OBJX) 
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/in34.f90 -o in34.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) in34.$(OBJX) fcheck.$(OBJX) $(LIBS) -o in34.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test in34
	in34.$(EXESUFFIX)

verify: ;

in34.run: run


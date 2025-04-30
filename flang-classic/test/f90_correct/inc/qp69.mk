#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test qtosint  ########


qp69: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp69.f08 fcheck.$(OBJX)
	-$(RM) qp69.$(EXESUFFIX) qp69.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp69.f08 -o qp69.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp69.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp69.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp69
	qp69.$(EXESUFFIX)

verify: ;

qp69.run: run


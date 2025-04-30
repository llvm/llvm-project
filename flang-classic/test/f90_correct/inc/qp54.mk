#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test holltoq  ########


qp54: run


fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp54.f08 fcheck.$(OBJX)
	-$(RM) qp54.$(EXESUFFIX) qp54.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp54.f08 -o qp54.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp54.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp54.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp54
	qp54.$(EXESUFFIX)

verify: ;

qp54.run: run


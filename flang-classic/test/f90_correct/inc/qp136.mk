#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#


########## Make rule for test qpowi  ########


qp136: run
	

fcheck.$(OBJX): $(SRC)/check_mod.F90
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o fcheck.$(OBJX)

build:  $(SRC)/qp136.f08 fcheck.$(OBJX)
	-$(RM) qp136.$(EXESUFFIX) qp136.$(OBJX)
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp136.f08 -o qp136.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp136.$(OBJX) fcheck.$(OBJX) $(LIBS) -o qp136.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp136
	qp136.$(EXESUFFIX)

verify: ;

qp136.run: run


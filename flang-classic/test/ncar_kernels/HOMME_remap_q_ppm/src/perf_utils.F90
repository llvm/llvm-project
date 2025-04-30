
! KGEN-generated Fortran source file
!
! Filename    : perf_utils.F90
! Generated at: 2015-02-24 15:34:48
! KGEN version: 0.4.4



    MODULE perf_utils
        !-----------------------------------------------------------------------
        !
        ! Purpose: This module supplies the csm_share and CAM utilities
        !          needed by perf_mod.F90 (when the csm_share and CAM utilities
        !          are not available).
        !
        ! Author:  P. Worley, October 2007
        !
        ! $Id$
        !
        !-----------------------------------------------------------------------
        !-----------------------------------------------------------------------
        !- module boilerplate --------------------------------------------------
        !-----------------------------------------------------------------------
        IMPLICIT NONE
        PRIVATE ! Make the default access private
        !
        ! Copyright (C) 2003-2014 Intel Corporation.  All Rights Reserved.
        !
        ! The source code contained or described herein and all documents
        ! related to the source code ("Material") are owned by Intel Corporation
        ! or its suppliers or licensors.  Title to the Material remains with
        ! Intel Corporation or its suppliers and licensors.  The Material is
        ! protected by worldwide copyright and trade secret laws and treaty
        ! provisions.  No part of the Material may be used, copied, reproduced,
        ! modified, published, uploaded, posted, transmitted, distributed, or
        ! disclosed in any way without Intel's prior express written permission.
        !
        ! No license under any patent, copyright, trade secret or other
        ! intellectual property right is granted to or conferred upon you by
        ! disclosure or delivery of the Materials, either expressly, by
        ! implication, inducement, estoppel or otherwise.  Any license under
        ! such intellectual property rights must be express and approved by
        ! Intel in writing.
        !      /* -*- Mode: Fortran; -*- */
        !
        !      (C) 2001 by Argonne National Laboratory.
        !
        !                                 MPICH2 COPYRIGHT
        !
        ! The following is a notice of limited availability of the code, and disclaimer
        ! which must be included in the prologue of the code and in all source listings
        ! of the code.
        !
        ! Copyright Notice
        !  + 2002 University of Chicago
        !
        ! Permission is hereby granted to use, reproduce, prepare derivative works, and
        ! to redistribute to others.  This software was authored by:
        !
        ! Mathematics and Computer Science Division
        ! Argonne National Laboratory, Argonne IL 60439
        !
        ! (and)
        !
        ! Department of Computer Science
        ! University of Illinois at Urbana-Champaign
        !
        !
        !                             GOVERNMENT LICENSE
        !
        ! Portions of this material resulted from work developed under a U.S.
        ! Government Contract and are subject to the following license: the Government
        ! is granted for itself and others acting on its behalf a paid-up, nonexclusive,
        ! irrevocable worldwide license in this computer software to reproduce, prepare
        ! derivative works, and perform publicly and display publicly.
        !
        !                                 DISCLAIMER
        !
        ! This computer code material was prepared, in part, as an account of work
        ! sponsored by an agency of the United States Government.  Neither the United
        ! States, nor the University of Chicago, nor any of their employees, makes any
        ! warranty express or implied, or assumes any legal liability or responsibility
        ! for the accuracy, completeness, or usefulness of any information, apparatus,
        ! product, or process disclosed, or represents that its use would not infringe
        ! privately owned rights.
        !
        ! Portions of this code were written by Microsoft. Those portions are
        ! Copyright (c) 2007 Microsoft Corporation. Microsoft grants permission to
        ! use, reproduce, prepare derivative works, and to redistribute to
        ! others. The code is licensed "as is." The User bears the risk of using
        ! it. Microsoft gives no express warranties, guarantees or
        ! conditions. To the extent permitted by law, Microsoft excludes the
        ! implied warranties of merchantability, fitness for a particular
        ! purpose and non-infringement.
        !
        !
        !
        !
        !
        !      DO NOT EDIT
        !      This file created by buildiface
        !
        !-----------------------------------------------------------------------
        ! Public interfaces ----------------------------------------------------
        !-----------------------------------------------------------------------

        !-----------------------------------------------------------------------
        ! Private interfaces ---------------------------------------------------
        !-----------------------------------------------------------------------
        !-----------------------------------------------------------------------
        !- include statements --------------------------------------------------
        !-----------------------------------------------------------------------
        !
        ! $Id: gptl.inc,v 1.44 2011-03-28 20:55:19 rosinski Exp $
        !
        ! Author: Jim Rosinski
        !
        ! GPTL header file to be included in user code. Values match
        ! their counterparts in gptl.h. See that file or man pages
        ! or web-based documenation for descriptions of each value
        !
        ! Externals
        !-----------------------------------------------------------------------
        ! Public data ---------------------------------------------------------
        !-----------------------------------------------------------------------
        !----------------------------------------------------------------------------
        ! precision/kind constants (from csm_share/shr/shr_kind_mod.F90)
        !----------------------------------------------------------------------------
        ! 8 byte real
        INTEGER, parameter, public :: shr_kind_i8 = selected_int_kind (13) ! 8 byte integer
        ! native integer
        ! long char
        ! extra-long char
        !-----------------------------------------------------------------------
        ! Private data ---------------------------------------------------------
        !-----------------------------------------------------------------------
        ! default
        ! unit number for log output
        !=======================================================================
        CONTAINS

        ! read subroutines
        !=======================================================================
        !
        !========================================================================
        !

        !============== Routines from csm_share/shr/shr_sys_mod.F90 ============
        !=======================================================================

        !===============================================================================
        !===============================================================================

        !===============================================================================
        !================== Routines from csm_share/shr/shr_mpi_mod.F90 ===============
        !===============================================================================

        !===============================================================================
        !===============================================================================

        !===============================================================================
        !===============================================================================

        !===============================================================================
        !===============================================================================

        !===============================================================================
        !===============================================================================

        !===============================================================================
        !================== Routines from csm_share/shr/shr_file_mod.F90 ===============
        !===============================================================================
        !BOP ===========================================================================
        !
        ! !IROUTINE: shr_file_getUnit -- Get a free FORTRAN unit number
        !
        ! !DESCRIPTION: Get the next free FORTRAN unit number.
        !
        ! !REVISION HISTORY:
        !     2005-Dec-14 - E. Kluzek - creation
        !     2007-Oct-21 - P. Worley - dumbed down for use in perf_mod
        !
        ! !INTERFACE: ------------------------------------------------------------------

        !===============================================================================
        !===============================================================================
        !BOP ===========================================================================
        !
        ! !IROUTINE: shr_file_freeUnit -- Free up a FORTRAN unit number
        !
        ! !DESCRIPTION: Free up the given unit number
        !
        ! !REVISION HISTORY:
        !     2005-Dec-14 - E. Kluzek - creation
        !     2007-Oct-21 - P. Worley - dumbed down for use in perf_mod
        !
        ! !INTERFACE: ------------------------------------------------------------------

        !===============================================================================
        !============= Routines from atm/cam/src/utils/namelist_utils.F90 ==============
        !===============================================================================

        !===============================================================================
        !================ Routines from atm/cam/src/utils/string_utils.F90 =============
        !===============================================================================

        !===============================================================================
    END MODULE perf_utils

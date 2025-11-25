import { createTheme, ThemeOptions } from '@mui/material/styles';

// Military-grade color palette
const militaryColors = {
  // Primary colors - Military green
  primary: {
    main: '#2E7D32',     // Military green
    dark: '#1B5E20',     // Dark green
    light: '#4CAF50',    // Light green
    contrastText: '#FFFFFF'
  },
  
  // Secondary colors - Warning amber
  secondary: {
    main: '#FFA726',     // Warning amber
    dark: '#F57C00',     // Dark amber
    light: '#FFB74D',    // Light amber
    contrastText: '#000000'
  },
  
  // Status colors
  error: {
    main: '#D32F2F',     // Danger red
    dark: '#C62828',     // Dark red
    light: '#F44336',    // Light red
    contrastText: '#FFFFFF'
  },
  
  warning: {
    main: '#FF9800',     // Alert orange
    dark: '#F57C00',     // Dark orange
    light: '#FFB74D',    // Light orange
    contrastText: '#000000'
  },
  
  info: {
    main: '#1976D2',     // Information blue
    dark: '#1565C0',     // Dark blue
    light: '#42A5F5',    // Light blue
    contrastText: '#FFFFFF'
  },
  
  success: {
    main: '#388E3C',     // Success green
    dark: '#2E7D32',     // Dark success
    light: '#4CAF50',    // Light success
    contrastText: '#FFFFFF'
  },
  
  // Custom clearance level colors
  clearance: {
    restricted: '#FFC107',      // Yellow
    confidential: '#FF9800',    // Orange
    secret: '#F44336',          // Red
    topSecret: '#9C27B0',       // Purple
    sci: '#3F51B5',             // Indigo
    sap: '#E91E63',             // Pink
    cosmic: '#000000',          // Black
  },
  
  // Risk level colors
  risk: {
    safe: '#4CAF50',         // Green
    low: '#8BC34A',          // Light green
    moderate: '#FFC107',     // Yellow
    high: '#FF9800',         // Orange
    critical: '#F44336',     // Red
    quarantined: '#9C27B0'   // Purple
  }
};

const militaryTheme: ThemeOptions = {
  palette: {
    mode: 'dark',
    primary: militaryColors.primary,
    secondary: militaryColors.secondary,
    error: militaryColors.error,
    warning: militaryColors.warning,
    info: militaryColors.info,
    success: militaryColors.success,
    
    background: {
      default: '#121212',      // Dark background
      paper: '#1E1E1E',        // Card background
    },
    
    text: {
      primary: '#FFFFFF',      // White text
      secondary: '#B0B0B0',    // Gray text
      disabled: '#666666'      // Disabled text
    },
    
    divider: '#333333',        // Divider lines
    
    action: {
      active: '#FFFFFF',
      hover: 'rgba(255, 255, 255, 0.08)',
      selected: 'rgba(255, 255, 255, 0.12)',
      disabled: 'rgba(255, 255, 255, 0.26)',
      disabledBackground: 'rgba(255, 255, 255, 0.12)'
    }
  },
  
  typography: {
    fontFamily: 'Roboto Mono, monospace',
    
    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
      letterSpacing: '0.02em',
      color: '#FFFFFF'
    },
    
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      letterSpacing: '0.01em',
      color: '#FFFFFF'
    },
    
    h3: {
      fontSize: '1.75rem',
      fontWeight: 600,
      color: '#FFFFFF'
    },
    
    h4: {
      fontSize: '1.5rem',
      fontWeight: 500,
      color: '#FFFFFF'
    },
    
    h5: {
      fontSize: '1.25rem',
      fontWeight: 500,
      color: '#FFFFFF'
    },
    
    h6: {
      fontSize: '1.1rem',
      fontWeight: 500,
      color: '#FFFFFF'
    },
    
    body1: {
      fontSize: '0.9rem',
      lineHeight: 1.6,
      color: '#FFFFFF'
    },
    
    body2: {
      fontSize: '0.8rem',
      lineHeight: 1.5,
      color: '#B0B0B0'
    },
    
    button: {
      fontWeight: 600,
      textTransform: 'uppercase',
      fontSize: '0.85rem',
      letterSpacing: '0.05em'
    },
    
    caption: {
      fontSize: '0.75rem',
      color: '#B0B0B0'
    },
    
    overline: {
      fontSize: '0.7rem',
      fontWeight: 600,
      textTransform: 'uppercase',
      letterSpacing: '0.1em',
      color: '#B0B0B0'
    }
  },
  
  shape: {
    borderRadius: 4
  },
  
  spacing: 8,
  
  components: {
    MuiAppBar: {
      styleOverrides: {
        root: {
          backgroundColor: '#1E1E1E',
          borderBottom: '2px solid #2E7D32'
        }
      }
    },
    
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 4,
          textTransform: 'uppercase',
          fontWeight: 600,
          minHeight: 40,
          transition: 'all 0.3s ease'
        },
        
        containedPrimary: {
          background: 'linear-gradient(45deg, #2E7D32 30%, #4CAF50 90%)',
          border: '1px solid #4CAF50',
          '&:hover': {
            background: 'linear-gradient(45deg, #1B5E20 30%, #2E7D32 90%)',
            boxShadow: '0 4px 8px rgba(46, 125, 50, 0.3)'
          },
          '&:active': {
            boxShadow: '0 2px 4px rgba(46, 125, 50, 0.5)'
          }
        },
        
        containedSecondary: {
          background: 'linear-gradient(45deg, #FF9800 30%, #FFA726 90%)',
          '&:hover': {
            background: 'linear-gradient(45deg, #F57C00 30%, #FF9800 90%)'
          }
        },
        
        containedError: {
          background: 'linear-gradient(45deg, #D32F2F 30%, #F44336 90%)',
          '&:hover': {
            background: 'linear-gradient(45deg, #C62828 30%, #D32F2F 90%)'
          }
        }
      }
    },
    
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundColor: '#1E1E1E',
          border: '1px solid #333333',
          borderRadius: 8,
          transition: 'all 0.3s ease',
          '&:hover': {
            borderColor: '#2E7D32',
            boxShadow: '0 4px 8px rgba(0, 0, 0, 0.3)'
          }
        }
      }
    },
    
    MuiCardHeader: {
      styleOverrides: {
        root: {
          backgroundColor: '#2A2A2A',
          borderBottom: '1px solid #333333'
        },
        title: {
          fontWeight: 600,
          fontSize: '1.1rem',
          color: '#FFFFFF'
        }
      }
    },
    
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 'bold',
          border: '2px solid',
          fontSize: '0.75rem'
        },
        
        filled: {
          '&.MuiChip-colorPrimary': {
            backgroundColor: militaryColors.primary.main,
            borderColor: militaryColors.primary.main,
            color: '#FFFFFF'
          }
        }
      }
    },
    
    MuiTextField: {
      styleOverrides: {
        root: {
          '& .MuiOutlinedInput-root': {
            '& fieldset': {
              borderColor: '#666666'
            },
            '&:hover fieldset': {
              borderColor: '#2E7D32'
            },
            '&.Mui-focused fieldset': {
              borderColor: '#4CAF50'
            }
          }
        }
      }
    },
    
    MuiAlert: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          fontWeight: 500
        },
        
        standardError: {
          backgroundColor: 'rgba(211, 47, 47, 0.2)',
          border: '1px solid #D32F2F',
          color: '#FFFFFF'
        },
        
        standardWarning: {
          backgroundColor: 'rgba(255, 152, 0, 0.2)',
          border: '1px solid #FF9800',
          color: '#FFFFFF'
        },
        
        standardInfo: {
          backgroundColor: 'rgba(25, 118, 210, 0.2)',
          border: '1px solid #1976D2',
          color: '#FFFFFF'
        },
        
        standardSuccess: {
          backgroundColor: 'rgba(56, 142, 60, 0.2)',
          border: '1px solid #388E3C',
          color: '#FFFFFF'
        }
      }
    },
    
    MuiDataGrid: {
      styleOverrides: {
        root: {
          backgroundColor: '#1E1E1E',
          border: '1px solid #333333',
          color: '#FFFFFF',
          
          '& .MuiDataGrid-columnHeaders': {
            backgroundColor: '#2A2A2A',
            borderBottom: '2px solid #2E7D32'
          },
          
          '& .MuiDataGrid-cell': {
            borderColor: '#333333'
          },
          
          '& .MuiDataGrid-row:hover': {
            backgroundColor: 'rgba(46, 125, 50, 0.1)'
          }
        }
      }
    }
  }
};

export const theme = createTheme(militaryTheme);

// Export color utilities
export { militaryColors };

// Helper functions for theme usage
export const getRiskLevelColor = (riskLevel: string): string => {
  switch (riskLevel.toUpperCase()) {
    case 'SAFE': return militaryColors.risk.safe;
    case 'LOW': return militaryColors.risk.low;
    case 'MODERATE': return militaryColors.risk.moderate;
    case 'HIGH': return militaryColors.risk.high;
    case 'CRITICAL': return militaryColors.risk.critical;
    case 'QUARANTINED': return militaryColors.risk.quarantined;
    default: return '#9E9E9E';
  }
};

export const getClearanceLevelColor = (clearanceLevel: string): string => {
  switch (clearanceLevel.toUpperCase()) {
    case 'RESTRICTED': return militaryColors.clearance.restricted;
    case 'CONFIDENTIAL': return militaryColors.clearance.confidential;
    case 'SECRET': return militaryColors.clearance.secret;
    case 'TOP_SECRET': return militaryColors.clearance.topSecret;
    case 'SCI': return militaryColors.clearance.sci;
    case 'SAP': return militaryColors.clearance.sap;
    case 'COSMIC': return militaryColors.clearance.cosmic;
    default: return '#9E9E9E';
  }
};

export default theme;